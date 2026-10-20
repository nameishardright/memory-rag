"""Download a pinned public multilingual model; never receives private text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
FILES = ['config.json', 'onnx/model_qint8_avx512_vnni.onnx', 'modules.json',
         'sentence_bert_config.json', 'sentencepiece.bpe.model',
         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
         '1_Pooling/config.json']


def default_model_dir():
    base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.local' / 'share')
    return base / 'memory-rag' / 'models' / 'multilingual-e5-small'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=default_model_dir())
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # curl honors this Windows machine's network configuration more reliably
    # than the Hub client's large-file transport. Requests contain no token.
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        raise RuntimeError('curl is required for the public model download')
    def download(url, path):
        subprocess.run([curl, '--silent', '--show-error', '--fail', '--location', '--retry', '2',
                        '--connect-timeout', '20', '--max-time', '1800',
                        '--continue-at', '-', '--output', str(path), url], check=True)
    def download_large(url, path, size):
        parts = path.with_name(path.name + '.parts')
        parts.mkdir(exist_ok=True)
        count = 16
        step = (size + count - 1) // count
        def part(i):
            left, right = i * step, min(size - 1, (i + 1) * step - 1)
            target = parts / ('part-' + str(i).zfill(2))
            expected_size = right - left + 1
            if target.is_file() and target.stat().st_size == expected_size:
                return target
            if target.is_file() and target.stat().st_size > expected_size:
                raise RuntimeError('Invalid partial model download size')
            # Preserve completed prefixes after a network interruption. Small
            # segments avoid repeatedly downloading a nearly complete 7 MB part.
            while (target.stat().st_size if target.is_file() else 0) < expected_size:
                completed = target.stat().st_size if target.is_file() else 0
                segment_left = left + completed
                segment_right = min(right, segment_left + 1024 * 1024 - 1)
                segment_size = segment_right - segment_left + 1
                segment = target.with_suffix('.segment')
                succeeded = False
                for endpoint in (url, url.replace('https://huggingface.co/', 'https://hf-mirror.com/')):
                    result = subprocess.run([curl, '--silent', '--show-error', '--fail', '--location',
                        '--retry', '2', '--retry-all-errors', '--retry-max-time', '360',
                        '--connect-timeout', '10', '--max-time', '120',
                        '--range', str(segment_left) + '-' + str(segment_right),
                        '--max-filesize', str(segment_size), '--output', str(segment),
                        '--write-out', '%{http_code}', endpoint], capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip() == '206' and segment.stat().st_size == segment_size:
                        succeeded = True
                        break
                if not succeeded:
                    raise RuntimeError('Model byte range unavailable; retry setup to continue')
                with target.open('ab') as output, segment.open('rb') as source:
                    shutil.copyfileobj(source, output)
                segment.unlink()
            return target
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(part, i) for i in range(count)]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                print('Public model byte ranges verified: ' + str(done) + '/' + str(count), flush=True)
        with path.open('wb') as output:
            for i in range(count):
                with (parts / ('part-' + str(i).zfill(2))).open('rb') as source:
                    shutil.copyfileobj(source, output)
    api = directory / 'model-api.download'
    if not api.is_file():
        download('https://huggingface.co/api/models/' + MODEL_ID + '/revision/' + REVISION + '?blobs=true', api)
    repository = json.loads(api.read_text(encoding='utf-8'))
    if repository.get('sha') != REVISION:
        raise RuntimeError('Unexpected model repository revision')
    entries = {row['rfilename']: row for row in repository['siblings']}
    hashes = {}
    for name in FILES:
        path = directory / name
        entry = entries[name]
        expected = entry.get('lfs', {}).get('sha256')
        if not path.is_file() or (expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_name(path.name + '.download')
            url = 'https://huggingface.co/' + MODEL_ID + '/resolve/' + REVISION + '/' + name
            if entry.get('size', 0) > 50 * 1024 * 1024:
                download_large(url, pending, entry['size'])
            else:
                download(url, pending)
            if entry.get('size') is not None and pending.stat().st_size != entry['size']:
                raise RuntimeError('Incomplete model download: ' + name)
            if expected and hashlib.sha256(pending.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Model checksum mismatch: ' + name)
            os.replace(pending, path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'backend': 'onnx_int8', 'sha256': hashes}
    (directory / 'model-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'ready': True, 'model_id': MODEL_ID, 'revision': REVISION,
                      'directory': str(directory), 'files_verified': len(hashes)}))


if __name__ == '__main__':
    main()
