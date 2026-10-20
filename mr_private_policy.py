"""Directory and purpose rules for the private corpus, independent of archives."""
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

POLICY_VERSION = 1
SCOPES = ('personal', 'technical', 'stock', 'history')
EXCLUDED_DIRS = frozenset({
    'node_modules', '.git', '.venv', 'venv', 'site-packages', '__pycache__',
    '.vs', 'bin', 'obj', 'dist', 'build', '.cache', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', '.next', '.nuxt', 'target',
    '_private-memory-rag', 'golden', 'evals', 'evaluation', 'benchmarks',
})
EXCLUDED_EXTENSIONS = frozenset({
    '.pyc', '.pyo', '.dll', '.exe', '.pdb', '.obj', '.lib', '.map', '.lock',
    '.suo', '.cache', '.bak', '.tmp', '.pending', '.pem', '.key', '.db', '.sqlite', '.sqlite3',
    '.ttf', '.woff', '.woff2', '.otf', '.eot', '.msi', '.iso', '.dmg', '.nupkg',
    '.class', '.jar',
})
CODE_EXTENSIONS = frozenset({
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.cs', '.cpp',
    '.c', '.h', '.hpp', '.java', '.go', '.ps1', '.bat', '.cmd', '.sh',
    '.css', '.scss', '.sql', '.sln', '.csproj', '.props', '.targets',
})
MEDIA_EXTENSIONS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.mp3', '.wav', '.m4a', '.flac', '.ogg'})
PERSONAL_EXTENSIONS = frozenset({
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.rtf',
    '.txt', '.md', '.markdown', '.csv', '.html', '.htm', '.json', '.ini',
    '.config', '.env', '.log', '.url', '.sqlite', '.db', '.png', '.jpg',
    '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif',
})
PROJECT_DIRS = frozenset({
    'projects', 'claude-relay-service', 'claude-skills', 'stock-agent',
    'job-workbench', 'life-agent', 'mycsharpproject', 'memory_rag',
    'memory-rag', 'dev-connect',
})
EVALUATION_NAME = re.compile(r'(?i)(?:^|[-_.])(?:eval(?:uation)?|benchmark|test-results|verification-results)(?:[-_.]|$)')


def load_policy(path=None):
    path = Path(path or os.environ.get('MR_PRIVATE_POLICY') or Path(__file__).with_name('private-policy.json'))
    value = {'version': POLICY_VERSION, 'rules': []}
    if path.is_file():
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if value.get('version') != POLICY_VERSION:
            raise ValueError('Unsupported private corpus policy version')
        for rule in value.get('rules', []):
            # A rule names either a path prefix or a directory found at any depth.
            if rule.get('scope') not in (*SCOPES, 'exclude') or bool(rule.get('prefix')) == bool(rule.get('component')):
                raise ValueError('Invalid private corpus policy rule')
    return value


def rule_matches(rule, parts):
    if rule.get('prefix'):
        prefix = normalized_parts(rule['prefix'])
        return parts[:len(prefix)] == prefix
    directories = parts[:-1]
    if rule.get('under'):
        parent = rule['under'].casefold()
        if parent not in directories:
            return False
        directories = directories[directories.index(parent) + 1:]
    pattern = rule['component'].casefold()
    return any(fnmatchcase(directory, pattern) for directory in directories)


def policy_signature(policy=None):
    value = {'algorithm': POLICY_VERSION, 'policy': policy or load_policy()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def normalized_parts(path):
    return tuple(part.casefold() for part in str(path).replace('\\', '/').replace('!/', '/').strip('/').split('/'))


def classify(path, dependency=False, text='', policy=None):
    parts = normalized_parts(path)
    name = parts[-1] if parts else ''
    suffix = PurePosixPath(name).suffix
    top = parts[0] if parts else ''
    purpose = 'knowledge'
    if any(PurePosixPath(owner).suffix.casefold() == '.jar' for owner in str(path).replace('\\', '/').split('!/')[:-1]):
        return {'scope': 'exclude', 'purpose': 'excluded', 'reason': 'compiled_java_dependency'}
    if dependency or EXCLUDED_DIRS.intersection(parts):
        return {'scope': 'exclude', 'purpose': 'excluded', 'reason': 'dependency_or_generated_directory'}
    if suffix in EXCLUDED_EXTENSIONS or name in {'desktop.ini', '.ds_store', 'package-lock.json', 'yarn.lock', '.env'} or name.startswith('.env.') or '.min.' in name:
        return {'scope': 'exclude', 'purpose': 'excluded', 'reason': 'generated_binary_or_secret_configuration'}
    if suffix == '.log' or (suffix == '.jsonl' and any(p in {'sessions', 'transcripts', 'rollouts', '.claude', '.codex'} for p in parts)):
        return {'scope': 'exclude', 'purpose': 'excluded', 'reason': 'raw_log_or_conversation'}
    frontmatter = re.match(r'\A\ufeff?---\s*\n(.*?)\n---(?:\s*\n|\s*$)', text[:2000], re.S)
    explicit_evaluation = bool(frontmatter and re.search(r'(?m)^rag_purpose:\s*evaluation\s*$', frontmatter.group(1)))
    if name.startswith('private_') and name.endswith(('_holdout.json', '_dev.json')):
        explicit_evaluation = True
    if (EVALUATION_NAME.search(name) and any(p.startswith(('eval', 'benchmark', 'verification')) for p in parts)) or explicit_evaluation:
        return {'scope': 'exclude', 'purpose': 'evaluation', 'reason': 'evaluation_material'}
    for rule in (policy or load_policy()).get('rules', []):
        if rule_matches(rule, parts):
            scope = rule['scope']
            return {'scope': scope, 'purpose': rule.get('purpose', 'excluded' if scope == 'exclude' else 'knowledge'),
                    'reason': rule.get('reason', 'directory_rule')}
    if top.startswith('90_') or 'backups' in parts or 'rollout_summaries' in parts:
        return {'scope': 'history', 'purpose': 'historical', 'reason': 'historical_backup_or_session_summary'}
    if any(p in {'股票分析', 'stock-analysis', 'stock_analysis', '股票资料'} for p in parts):
        return {'scope': 'stock', 'purpose': purpose, 'reason': 'stock_topic'}
    imported_project = top == '导入' and any(p in {'海马云', 'haimayun', 'runninghub'} for p in parts)
    if top.startswith('06_') or top in {'service etree db', 'vs code'} or PROJECT_DIRS.intersection(parts) or imported_project:
        return {'scope': 'technical', 'purpose': purpose, 'reason': 'project_directory'}
    if suffix in CODE_EXTENSIONS:
        return {'scope': 'technical', 'purpose': purpose, 'reason': 'source_code'}
    if top.startswith(('01_', '02_', '03_', '04_', '05_')) or top.startswith('@onenote') or suffix in PERSONAL_EXTENSIONS | MEDIA_EXTENSIONS:
        return {'scope': 'personal', 'purpose': purpose, 'reason': 'personal_document_or_curated_note'}
    return {'scope': 'technical', 'purpose': purpose, 'reason': 'other_supported_material'}


def project_group(path):
    original = tuple(str(path).replace('\\', '/').split('/'))
    parts = tuple(p.casefold() for p in original)
    if not parts:
        return None
    if parts[0].startswith('06_') and len(parts) > 2:
        return '/'.join(original[:2])
    for i, part in enumerate(parts):
        if part in PROJECT_DIRS or (parts[0] == '导入' and part in {'海马云', 'haimayun', 'runninghub'}):
            end = i + (2 if part == 'projects' and len(parts) > i + 2 else 1)
            return '/'.join(original[:end])
    return None
