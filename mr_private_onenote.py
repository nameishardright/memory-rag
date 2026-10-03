"""Read linked OneDrive notebooks through the local OneNote application."""
import hashlib
from pathlib import Path
import re
import urllib.parse
from xml.etree import ElementTree as ET

from mr_private import append_journal, read_journal


def application():
    # Use the registered vtable interface directly. Some Office installations
    # have a working OneNote server but no usable automation type library.
    import comtypes.client
    from comtypes import COMMETHOD, GUID, HRESULT
    from comtypes.automation import BSTR, IDispatch
    from ctypes import POINTER, c_int
    class IOneNote(IDispatch):
        _iid_ = GUID('{452AC71A-B655-4967-A208-A4CC39DD7949}')
        _methods_ = [
            COMMETHOD([], HRESULT, 'GetHierarchy', (['in'], BSTR, 'start'),
                      (['in'], c_int, 'scope'), (['out'], POINTER(BSTR), 'xml'), (['in'], c_int, 'schema')),
            COMMETHOD([], HRESULT, 'UpdateHierarchy', (['in'], BSTR, 'xml'), (['in'], c_int, 'schema')),
            COMMETHOD([], HRESULT, 'GetPageContent', (['in'], BSTR, 'page'),
                      (['out'], POINTER(BSTR), 'xml'), (['in'], c_int, 'info'), (['in'], c_int, 'schema')),
        ]
    return comtypes.client.CreateObject('OneNote.Application', interface=IOneNote)


def page_text(xml):
    from lxml import html
    tree = ET.fromstring(xml)
    lines = []
    for node in tree.iter():
        if node.tag.rsplit('}', 1)[-1] in {'T', 'OCRText'} and node.text:
            try:
                text = html.fragment_fromstring(node.text, create_parent=True).text_content()
            except (ValueError, ET.ParseError):
                text = node.text
            if text.strip(): lines.append(text.strip())
    return '\n\n'.join(lines)


def _links(root):
    entries = []
    for category in root.iterdir():
        if not category.is_dir() or not category.name.startswith('05_'):
            continue
        for path in category.rglob('*.url'):
            raw = path.read_bytes()
            encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = raw.decode(encoding, errors='replace')
            match = re.search(r'(?im)^URL=(.+)$', text)
            if not match:
                continue
            target = match.group(1).strip()
            if 'onedrive.live.com' not in target and 'd.docs.live.net' not in target:
                continue
            entries.append((path, target))
    return entries


def matching_notebooks(notebooks, link, target):
    target = target.removeprefix('onenote:')
    parsed = urllib.parse.urlsplit(target)
    query = urllib.parse.parse_qs(parsed.query)
    resid = (query.get('resid') or [''])[0].casefold()
    cid = resid.split('!')[0]
    target_path = urllib.parse.unquote(parsed.path).rstrip('/').casefold()
    exact, named = [], []
    for notebook in notebooks:
        location = notebook.attrib.get('path', '').removeprefix('onenote:')
        notebook_url = urllib.parse.urlsplit(location)
        location_path = urllib.parse.unquote(notebook_url.path).rstrip('/').casefold()
        same_path = (parsed.hostname and parsed.hostname == notebook_url.hostname and target_path
                     and (target_path == location_path or target_path.startswith(location_path + '/')))
        notebook_resid = (urllib.parse.parse_qs(notebook_url.query).get('resid') or [''])[0].casefold()
        if same_path or (resid and resid == notebook_resid):
            exact.append(notebook)
        elif notebook.attrib.get('name', '').casefold() == link.stem.casefold() and (not cid or cid in location.casefold()):
            named.append(notebook)
    selected = exact or named
    # A shared account ID alone does not identify the linked notebook.
    return selected if len(selected) == 1 else []


def harvest(root, journal, password):
    root, journal = Path(root), Path(journal)
    links = _links(root)
    if not links:
        return {'state': 'no_linked_notebooks', 'linked_notebooks': 0, 'pages': 0}
    app = application()
    tree = ET.fromstring(app.GetHierarchy('', 4, 2))
    notebooks = [n for n in tree.iter() if n.tag.rsplit('}', 1)[-1] == 'Notebook']
    previous = {}
    if journal.is_file():
        for row in read_journal(journal, password):
            if row.get('record_type') == 'archive_member': previous[row['path']] = row.get('source_sha256')
    records, total, failures, matched, inventories = [], 0, 0, 0, []
    from mr_private import now
    for link, target in links:
        selected = matching_notebooks(notebooks, link, target)
        if not selected:
            continue
        matched += 1
        owner = str(link.relative_to(root))
        paths = []
        for notebook in selected:
            for page in notebook.iter():
                if page.tag.rsplit('}', 1)[-1] != 'Page': continue
                page_id = page.attrib['ID']
                path = owner + '!/OneNote/' + hashlib.sha256(page_id.encode()).hexdigest()[:24] + '.md'
                paths.append(path)
                total += 1
                try:
                    xml = app.GetPageContent(page_id, 0, 2)
                    digest = hashlib.sha256(xml.encode()).hexdigest()
                    if previous.get(path) == digest: continue
                    text = '# ' + page.attrib.get('name', 'OneNote') + '\n\n' + page_text(xml)
                    records.append({'record_type': 'archive_member', 'archive': owner, 'path': path,
                                    'read_ok': True, 'text': text, 'method': 'onenote_com',
                                    'sha256': digest, 'source_sha256': digest, 'dependency': False,
                                    'extractor_version': 3, 'page_id': page_id,
                                    'source_updated_at': page.attrib.get('lastModifiedTime'), 'warnings': []})
                except Exception:
                    failures += 1
        inventories.append({'record_type': 'external_inventory', 'time': now(), 'owner': owner, 'paths': paths})
    if records or inventories:
        append_journal(journal, password, [*records, *inventories])
    return {'state': 'updated' if matched else 'notebook_not_open', 'linked_notebooks': len(links),
            'matched_notebooks': matched, 'pages': total, 'new_or_changed_pages': len(records), 'failed_pages': failures}
