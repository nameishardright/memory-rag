"""Stage public GitHub snapshots and audit their inclusion after a local unlock."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import os
import zipfile

PUBLIC_REPOS = {'memory-rag', 'nameishardright', 'ood-object-oriented-design', 'presentations'}
GITHUB_DIRECTORY = '06_代码项目/GitHub资料快照'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stage_github_snapshots(archives, onedrive):
    archives, onedrive = Path(archives).resolve(), Path(onedrive).resolve()
    destination = (onedrive / GITHUB_DIRECTORY).resolve()
    if not destination.is_relative_to(onedrive):
        raise ValueError('GitHub staging destination must stay inside OneDrive')
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(archives.glob('*.zip')):
        match = re.fullmatch(r'(.+)-([0-9a-f]{12,40})\.zip', path.name)
        if not match or match.group(1).casefold() not in PUBLIC_REPOS:
            continue
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Public GitHub ZIP integrity check failed')
        digest = sha256(path)
        target = destination / path.name
        if target.exists():
            if sha256(target) != digest:
                raise ValueError('Existing GitHub snapshot has different content')
        else:
            fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.pending', dir=destination)
            os.close(fd)
            try:
                shutil.copyfile(path, temporary)
                if sha256(temporary) != digest:
                    raise ValueError('GitHub copy hash mismatch')
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        entries.append({'repository': 'nameishardright/' + match.group(1),
                        'commit_prefix': match.group(2), 'archive': path.name,
                        'sha256': digest, 'bytes': path.stat().st_size})
    if not entries:
        raise ValueError('No recognized public GitHub snapshots')
    manifest = {'version': 1, 'source_kind': 'public_github_snapshot',
                'live_sync': False, 'scope': 'technical', 'archives': entries}
    (destination / 'github-source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {'archives': len(entries), 'bytes': sum(e['bytes'] for e in entries),
            'sha256_verified': True, 'destination': str(destination), 'indexed': False}


def audit_sources(rag):
    rag.ensure_scope('technical')
    github, ai_backup, scoped = 0, 0, 0
    for document in rag.documents.values():
        if document.get('purpose') in {'navigation', 'evaluation', 'excluded'}:
            continue
        names = [document['file'], *[alias['file'] for alias in document.get('duplicate_sources', [])]]
        if any(name.replace('\\', '/').startswith(GITHUB_DIRECTORY + '/') and '!/' in name for name in names):
            github += 1
            scoped += document['scope'] == 'technical'
        if any('AI备份' in name.replace('\\', '/').split('/') for name in names):
            ai_backup += 1
    return {'github_snapshot_documents': github, 'github_documents_in_technical_scope': scoped,
            'onedrive_ai_backup_documents': ai_backup,
            'audited_scopes': sorted(rag.loaded_scopes),
            'github_snapshot_content_confirmed': github > 0,
            'onedrive_ai_backup_content_confirmed': ai_backup > 0,
            'boundary': 'Document inclusion, not proof that every cloud source or current GitHub commit is covered.'}
