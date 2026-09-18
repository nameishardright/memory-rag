"""Run the offline suite and record counts plus the tested file hashes."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(CODE.glob("*.py")) + sorted((CODE / "tests").glob("*.py")) + sorted((CODE / "scripts").glob("*.py")) + [CODE / "corpus-policy.json"]
    hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    suite = unittest.defaultTestLoader.discover(str(CODE / "tests"))
    log = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(CODE))] for p in files)
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passed": result.wasSuccessful() and unchanged,
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "seconds": round(time.perf_counter() - start, 3), "files_sha256": hashes,
              "files_unchanged_during_run": unchanged, "log": log.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"files_sha256", "log"}}, ensure_ascii=False))
    if not report["passed"]:
        print(log.getvalue())
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
