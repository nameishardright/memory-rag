"""Native local UI. Unlock in memory, return evidence, and lock after inactivity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk

from mr_index import BM25
from mr_private import default_store, index_cache_path, load_private_rag, load_store
from mr_private_facts import CARD_INTENT, compact_search, extract_cards


class PrivateMemoryApp:
    def __init__(self, root, store):
        self.root, self.store = root, Path(store)
        self.password = None
        self.rag = None
        self.cards = None
        self.loaded_signature = None
        self.pending_search = False
        self.generation = 0
        self.rows = {}
        self.messages = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_activity = time.monotonic()
        self.lock_seconds = 600
        self.preparing = False
        root.title('我的加密记忆')
        root.geometry('1100x760')
        root.minsize(820, 580)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind_all('<KeyPress>', self.activity)
        root.bind_all('<ButtonPress>', self.activity)
        root.bind_all('<MouseWheel>', self.activity)

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='我的加密记忆', font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='在本机解锁和检索 · 原文证据随结果提供 · 空闲 10 分钟自动锁定').pack(anchor='w', pady=(2, 12))

        unlock = ttk.Frame(outer)
        unlock.pack(fill='x')
        ttk.Label(unlock, text='档案密码').pack(side='left')
        self.password_entry = ttk.Entry(unlock, show='●', width=28)
        self.password_entry.pack(side='left', padx=8)
        self.password_entry.bind('<Return>', lambda event: self.unlock())
        self.unlock_button = ttk.Button(unlock, text='解锁', command=self.unlock)
        self.unlock_button.pack(side='left')
        ttk.Button(unlock, text='立即锁定', command=self.lock).pack(side='left', padx=8)
        self.reload_button = ttk.Button(unlock, text='加载最新资料', command=self.reload, state='disabled')
        self.reload_button.pack(side='right')

        search = ttk.Frame(outer)
        search.pack(fill='x', pady=(16, 8))
        self.query = ttk.Entry(search)
        self.query.pack(side='left', fill='x', expand=True)
        self.query.bind('<Return>', lambda event: self.search())
        self.search_button = ttk.Button(search, text='查找', command=self.search, state='disabled')
        self.search_button.pack(side='left', padx=(8, 0))
        self.all_sources = tk.BooleanVar(value=False)
        self.scope_labels = {'自动选择': 'auto', '个人资料': 'personal', '代码项目': 'technical',
                             '股票资料': 'stock', '历史资料': 'history', '全部资料': 'all'}
        self.scope_choice = tk.StringVar(value='自动选择')
        ttk.Combobox(search, textvariable=self.scope_choice, values=list(self.scope_labels),
                     state='readonly', width=12).pack(side='left', padx=8)
        shortcuts = ttk.Frame(outer)
        shortcuts.pack(fill='x')
        for label in ('信用卡', '家庭关系', '旅行', '护照', '地址', '账号'):
            ttk.Button(shortcuts, text=label, command=lambda text=label: self.search_for(text, '个人资料')).pack(side='left', padx=(0, 6))
        ttk.Button(shortcuts, text='清理建议', command=self.cleanup_review).pack(side='left', padx=(0, 6))
        ttk.Button(shortcuts, text='收录缺口', command=self.coverage_review).pack(side='left', padx=(0, 6))
        self.prepare_button = ttk.Button(shortcuts, text='准备语义检索', command=self.prepare_semantic, state='disabled')
        self.prepare_button.pack(side='left', padx=(0, 6))

        self.status = tk.StringVar(value='输入原档案密码解锁。数据以加密文件保存在 OneDrive。')
        ttk.Label(outer, textvariable=self.status, wraplength=1040).pack(anchor='w', pady=12)
        self.tree = ttk.Treeview(outer, columns=('label', 'value', 'source'), show='headings', height=12)
        for column, label, width in [('label', '记录 / 段落', 240), ('value', '内容', 330), ('source', '来源', 430)]:
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, minwidth=90)
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.select)
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x', pady=8)
        ttk.Button(buttons, text='查看完整来源', command=self.full_source).pack(side='left')
        ttk.Button(buttons, text='复制所选结果', command=self.copy_selected).pack(side='left', padx=8)
        ttk.Label(buttons, text='历史记录不等于当前状态；OCR 内容请核对原件。').pack(side='right')
        self.detail = tk.Text(outer, height=10, wrap='word', font=('Microsoft YaHei UI', 10))
        self.detail.pack(fill='both', expand=True)
        self.detail.configure(state='disabled')
        self.password_entry.focus_set()
        root.after(100, self.poll)
        root.after(1000, self.tick)

    def activity(self, event=None):
        self.last_activity = time.monotonic()

    def write_detail(self, text):
        self.detail.configure(state='normal')
        self.detail.delete('1.0', 'end')
        self.detail.insert('1.0', text)
        self.detail.configure(state='disabled')

    def submit(self, kind, function):
        generation = self.generation
        def worker():
            try:
                self.messages.put((generation, kind, function(), None))
            except Exception as error:
                self.messages.put((generation, kind, None, type(error).__name__))
        self.executor.submit(worker)

    def unlock(self):
        if self.password is not None:
            return
        password = self.password_entry.get()
        self.password_entry.delete(0, 'end')
        if not password:
            return
        self.unlock_button.configure(state='disabled')
        self.status.set('正在解锁并准备检索……')
        self.password = password
        self.submit('unlock', lambda: self.load_snapshot(password))

    def signature(self):
        stat = self.store.stat()
        parts = [(stat.st_size, stat.st_mtime_ns)]
        for path in (index_cache_path(self.store, 'personal', 'semantic'),
                     self.store.parent / 'retrieval-settings.json'):
            try:
                extra = path.stat(); parts.append((extra.st_size, extra.st_mtime_ns))
            except FileNotFoundError:
                parts.append(None)
        return tuple(parts)

    def load_snapshot(self, password):
        signature = self.signature()
        return load_private_rag(self.store, password), signature

    def reload(self, then_search=False):
        if self.password is None:
            return
        password = self.password
        self.status.set('正在加载最新的加密资料……')
        self.pending_search = then_search
        self.reload_button.configure(state='disabled')
        self.search_button.configure(state='disabled')
        self.submit('unlock', lambda: self.load_snapshot(password))

    def search_for(self, text, scope=None):
        if scope is not None:
            self.scope_choice.set(scope)
            self.all_sources.set(False)
        self.query.delete(0, 'end')
        self.query.insert(0, text)
        self.search()

    def search(self):
        if self.rag is None:
            self.status.set('请先输入档案密码解锁。')
            return
        query = self.query.get().strip()
        if not query:
            return
        try:
            if self.signature() != self.loaded_signature:
                self.reload(then_search=True)
                return
        except OSError:
            self.status.set('无法读取最新索引，请等待 OneDrive 同步完成后重试。')
            return
        self.activity()
        self.status.set('正在本机检索……')
        self.search_button.configure(state='disabled')
        rag = self.rag
        scope = 'all' if self.all_sources.get() else self.scope_labels[self.scope_choice.get()]
        from mr_private_routing import route_query
        resolved = route_query(query)['scope'] if scope == 'auto' else scope
        if resolved == 'personal' and CARD_INTENT.search(query):
            cards = self.cards
            self.submit('cards', lambda: cards if cards is not None else extract_cards(rag.payload['documents']))
        else:
            self.submit('search', lambda: compact_search(rag, query, topk=5, scope=scope, preview_chars=700))

    def prepare_semantic(self):
        if self.rag is None or self.preparing:
            return
        scope = self.scope_labels[self.scope_choice.get()]
        if scope in {'auto', 'all'}:
            scope = 'personal'
        self.preparing = True
        self.prepare_button.configure(state='disabled')
        self.search_button.configure(state='disabled')
        self.status.set('正在本机准备语义索引，首次需要一些时间；可随时锁定停止。')
        rag, generation = self.rag, self.generation
        def progress(done, total):
            self.messages.put((generation, 'prepare_progress', {'done': done, 'total': total}, None))
        self.submit('prepare', lambda: rag.prepare_semantic(scope, progress))

    def show_coverage(self):
        payload = self.rag.payload
        coverage = payload['coverage']
        self.status.set('已解锁。默认库 {} 份资料；应收文件 {} 个，尚未处理 {} 个。索引更新：{}。'.format(
            coverage.get('default_documents', coverage['documents']),
            coverage.get('eligible_files', coverage['visible_files']),
            coverage.get('eligible_not_processed', coverage.get('not_processed', 0)), payload['updated_at'][:19]))

    def coverage_review(self):
        if self.password is None:
            self.status.set('请先解锁资料。')
            return
        password = self.password
        path = self.store.parent / 'coverage-review.enc'
        if not path.is_file():
            self.status.set('收录核对清单将在下一次索引更新后生成。')
            return
        self.submit('coverage', lambda: load_store(path, password))

    def show_coverage_review(self, report):
        labels = {'password_locked_pdf': 'PDF 需要原文件密码', 'web_notebook_link_only': '网页笔记入口',
                  'media_transcription_failed': '语音转写未完成', 'local_speech_transcript': '语音转写待核对',
                  'media_no_audio': '视频没有音轨', 'media_no_speech': '未识别到讲话',
                  'empty': '没有可检索正文', 'read_failed': '源文件读取失败',
                  'extraction_failed': '正文提取失败', 'legacy_office_partial_strings': '旧版 Office 文字不完整',
                  'source_changed_since_extraction': '源文件已修改，等待重新读取',
                  'archive_pending': '压缩包待展开', 'archive_index_failed': '压缩包内容读取失败'}
        labels.update(password_locked_archive_member='压缩包内文件需要原密码',
                      archive_member_read_failed='压缩包内文件读取失败',
                      archive_member_not_extracted='压缩包内文件尚未提取正文')
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self.write_detail('')
        entries = [(item.get('path', ''), labels.get(item.get('method'), '正文或来源需要核对'),
                    item.get('warnings', [])) for item in report.get('gaps', [])]
        entries.extend((path, '尚未处理', []) for path in report.get('not_processed', []))
        for i, (path, label, warnings) in enumerate(entries):
            key = str(i)
            excerpt = label + '\n\n来源：' + path
            if warnings: excerpt += '\n\n核对说明：' + '\n'.join(map(str, warnings))
            self.rows[key] = {'kind': 'cleanup', 'excerpt': excerpt}
            self.tree.insert('', 'end', iid=key, values=(label, '', path))
        self.status.set('资料核对清单：{} 项。目录排除项不计作漏收。生成时间：{}。'.format(
            len(entries), report.get('updated_at', '')[:19]))

    def cleanup_review(self):
        if self.password is None or self.rag is None:
            self.status.set('请先输入档案密码解锁。')
            return
        path = self.store.parent / 'cleanup-review.enc'
        if not path.is_file():
            self.status.set('尚未生成清理核对清单。')
            return
        password = self.password
        self.activity()
        self.status.set('正在解锁清理核对清单……')
        self.submit('cleanup', lambda: load_store(path, password))

    def show_cleanup(self, report):
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self.write_detail('')
        def add(label, value, source, detail):
            key = str(len(self.rows))
            self.rows[key] = {'kind': 'cleanup', 'excerpt': detail}
            self.tree.insert('', 'end', iid=key, values=(label, value, source))
        size = lambda value: '{:.2f} MiB'.format(value / (1024 * 1024))
        for group in report['cache_candidates']:
            detail = ('可能可重建的项目目录：\n' + group['directory'] + '\n大小：' + size(group['bytes']) +
                      '\n文件数：' + str(group['files']) + '\n\n恢复依据：\n' +
                      '\n'.join(group['restore_manifest_candidates']) +
                      '\n\n须确认源码完整、依赖仍可下载、目录中没有手工保存的资料。目录名称本身不构成可删证明。')
            add('项目缓存 / 构建产物（待确认）', size(group['bytes']), group['directory'], detail)
        groups = sorted(report['exact_duplicates'], key=lambda group: (group['contains_cache_files'], -group['potential_redundant_bytes']))
        for group in groups:
            paths = '\n'.join(file['path'] for file in group['files'])
            detail = ('以下文件已核对完整 SHA-256 相同：\n' + paths + '\n\nSHA-256：' + group['sha256'] +
                      '\n保留一份时的多余内容大小：' + size(group['potential_redundant_bytes']) +
                      '\n\n保留位置由你决定。删除前须重新核对文件，并检查项目、资料包或时间线是否引用各副本。')
            if group['contains_cache_files']:
                detail += '\n涉及程序依赖：不宜单独删除重复的依赖文件，应按完整项目目录决定。'
            add('哈希相同（{} 份）'.format(group['copies']), size(group['potential_redundant_bytes']), group['keep_reference'], detail)
        for group in report['same_name_different']:
            detail = '同名但内容不同，不可当重复文件删除：\n' + '\n'.join(file['path'] for file in group['files'])
            add('同名但不同：保留核对', group['name'], group['files'][0]['path'], detail)
        for file in report['largest_files']:
            detail = '大文件：\n' + file['path'] + '\n大小：' + size(file['size']) + '\n\n体积大不代表不重要；本项没有证据说明它可删除。'
            add('大文件（仅供查看）', size(file['size']), file['path'], detail)
        summary = report['summary']
        self.status.set('清单时间 {}；哈希已核对 {}/{} 个文件；重复组 {}。本页面只供核对。大小按逻辑内容计算，缓存与重复项可能重叠。'.format(
            report['created_at'][:19], summary['hash_verified_files'], summary['visible_files'], summary['exact_duplicate_groups']))

    def poll(self):
        try:
            while True:
                generation, kind, result, error = self.messages.get_nowait()
                if generation != self.generation:
                    continue
                if error:
                    if kind == 'unlock':
                        self.lock()
                        self.status.set('解锁失败：请检查密码、文件是否已下载，以及加密文件是否完整。')
                    else:
                        if kind == 'prepare':
                            self.preparing = False
                            self.prepare_button.configure(state='normal')
                        self.search_button.configure(state='normal')
                        self.status.set('检索未完成，请重试。错误类型：' + error)
                    continue
                if kind == 'unlock':
                    previous = self.rag
                    self.rag, self.loaded_signature = result
                    if previous is not None and previous is not self.rag:
                        previous.close()
                    self.cards = None
                    self.rows.clear()
                    self.tree.delete(*self.tree.get_children())
                    self.write_detail('')
                    self.reload_button.configure(state='normal')
                    self.search_button.configure(state='normal')
                    self.prepare_button.configure(state='normal')
                    self.show_coverage()
                    self.query.focus_set()
                    if self.pending_search:
                        self.pending_search = False
                        self.search()
                elif kind == 'cleanup':
                    self.show_cleanup(result)
                elif kind == 'coverage':
                    self.show_coverage_review(result)
                elif kind == 'prepare_progress':
                    self.status.set('正在准备语义检索：{}/{} 个段落窗口。资料只在本机处理。'.format(result['done'], result['total']))
                elif kind == 'prepare':
                    self.preparing = False
                    self.prepare_button.configure(state='normal')
                    self.search_button.configure(state='normal')
                    self.status.set('语义检索已准备好：{} 个窗口，新计算 {} 个，复用 {} 个。'.format(
                        result['windows'], result['encoded'], result['reused']))
                elif kind in {'cards', 'search'}:
                    self.rows.clear()
                    self.tree.delete(*self.tree.get_children())
                    self.write_detail('')
                    if kind == 'cards':
                        self.cards = result
                        for n, card in enumerate(result):
                            key = str(n)
                            self.rows[key] = {'kind': 'card', **card}
                            sources = '; '.join(s['file'] + ':' + str(s['line']) for s in card['sources'])
                            self.tree.insert('', 'end', iid=key, values=(card['label'], card['number'], sources))
                        self.status.set('找到 {} 条去重卡号记录。卡名来自原文，卡种和当前有效状态仍需核对。'.format(len(result)))
                    else:
                        for n, hit in enumerate(result['hits']):
                            key = str(n)
                            self.rows[key] = {'kind': 'passage', **hit}
                            self.tree.insert('', 'end', iid=key, values=(hit['heading'], hit['excerpt'][:160], hit['file']))
                        labels = {value: key for key, value in self.scope_labels.items()}
                        mode_label = '关键词＋语义' if result['mode'] == 'hybrid' else '关键词'
                        hint = '；语义索引尚未就绪，可点击“准备语义检索”' if result['mode'] == 'bm25(fallback)' else ''
                        route = result.get('route', {})
                        searched = '＋'.join(labels.get(scope, scope) for scope in
                                            [result['scope'], *route.get('also_searched', [])])
                        if route.get('deferred_scope'):
                            hint += '；“{}”较大，未自动加载，需要时请在范围里选择它'.format(
                                labels.get(route['deferred_scope'], route['deferred_scope']))
                        self.status.set('找到 {} 条相关来源 · {} · {}{}。相关资料仍需核对人物、日期与事实。'.format(
                            len(result['hits']), searched, mode_label, hint))
                    self.search_button.configure(state='normal')
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def selected(self):
        selection = self.tree.selection()
        return self.rows.get(selection[0]) if selection else None

    def select(self, event=None):
        row = self.selected()
        if not row:
            return
        if row['kind'] == 'cleanup':
            self.write_detail(row['excerpt'])
        elif row['kind'] == 'card':
            sources = '\n'.join(s['file'] + '，提取文本第 ' + str(s['line']) + ' 行' for s in row['sources'])
            self.write_detail(row['label'] + '\n' + row['number'] + '\n\n来源：\n' + sources +
                              '\n\n号码已通过格式校验；这不代表卡片仍有效，也不能单凭格式认定为信用卡。')
        else:
            evidence = row.get('evidence', {})
            text = evidence.get('context', row['excerpt'])
            warnings = evidence.get('warnings') or []
            reasons = evidence.get('review_reasons') or []
            metadata = '\n正文提取：' + str(evidence.get('method') or '原文')
            if warnings:
                metadata += '\n提取警告：' + '；'.join(map(str, warnings))
            if evidence.get('source_updated_at'):
                metadata += '\n源文件更新时间：' + str(evidence['source_updated_at']) + '（不是事实生效日期）'
            if reasons:
                labels = {'requested_year_not_in_evidence': '证据中未找到所问年份',
                          'source_extraction_requires_review': '提取文字需核对原件',
                          'navigation_or_historical_source': '目录或历史资料不能直接证明当前事实',
                          'current_status_requires_confirmation': '当前状态需要有效记录确认',
                          'requested_person_role_not_in_evidence': '证据中未找到所问的家庭成员身份',
                          'requested_appointment_not_in_evidence': '证据中未找到所问预约',
                          'requested_birth_date_not_recorded': '证据中未找到明确的出生日期记录',
                          'draft_or_third_party_estimate': '草稿、目标或市场估计不能直接当作个人事实'}
                metadata += '\n核对事项：' + '；'.join(labels.get(r, r) for r in reasons)
            self.write_detail(row['file'] + metadata + '\n\n' + text +
                              ('\n\n［已显示相关段落；可查看完整来源。］' if not evidence.get('context_complete') else ''))

    def full_source(self):
        row = self.selected()
        if not row or self.rag is None:
            return
        if row['kind'] == 'cleanup':
            self.write_detail(row['excerpt'])
            return
        doc_id = row['sources'][0]['document_id'] if row['kind'] == 'card' else row['document_id']
        doc = self.rag.get(doc_id)
        if doc.get('ok'):
            self.write_detail(doc['file'] + '\n\n' + doc['text'])

    def copy_selected(self):
        row = self.selected()
        if not row:
            return
        if row['kind'] == 'passage':
            self.select()
            text = self.detail.get('1.0', 'end-1c')
        else:
            text = row['number'] if row['kind'] == 'card' else row['excerpt']
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.activity()
        self.status.set('已复制所选结果。')

    def lock(self):
        self.generation += 1
        if self.rag is not None:
            self.rag.close()
        self.password = self.rag = self.cards = None
        self.preparing = False
        self.loaded_signature = None
        self.pending_search = False
        self.rows.clear()
        BM25._TOK_CACHE.clear()
        self.tree.delete(*self.tree.get_children())
        self.write_detail('')
        self.password_entry.delete(0, 'end')
        self.query.delete(0, 'end')
        self.unlock_button.configure(state='normal')
        self.search_button.configure(state='disabled')
        self.reload_button.configure(state='disabled')
        self.prepare_button.configure(state='disabled')
        self.status.set('已锁定。输入密码重新解锁。')
        self.password_entry.focus_set()

    def tick(self):
        if self.password is not None and time.monotonic() - self.last_activity >= self.lock_seconds:
            self.lock()
        self.root.after(1000, self.tick)

    def close(self):
        self.lock()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch(store=None):
    root = tk.Tk()
    PrivateMemoryApp(root, store or default_store())
    root.mainloop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=default_store())
    launch(parser.parse_args().store)
