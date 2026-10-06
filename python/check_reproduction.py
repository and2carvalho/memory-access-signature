"""Regenerates the published results and compares them with the versioned files.

Two independent implementations are checked against output/atlas.json and
output/crossdomain.json:

  * the TypeScript reference pipeline (src/*.ts), unless --skip-ts is given;
  * the numba kernels of the Python port (memsig/fast.py).

Rows are compared field by field after keying them by their experimental cell, so the
check is insensitive to row order and to the `generatedAt` timestamp only.

Usage (from the repository root):
    python python/check_reproduction.py [--skip-ts]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

KEYS = {"atlas": ("algo", "pattern", "n"), "crossdomain": ("workload", "n")}


def normalize(value):
    """Converts numpy scalars and nested containers to plain Python values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def compare(name, reference, candidate, label):
    reference, candidate = normalize(reference), normalize(candidate)
    problems = []
    for field in sorted(set(reference) | set(candidate)):
        if field in ("generatedAt", "rows"):
            continue
        if reference.get(field) != candidate.get(field):
            problems.append(f"field '{field}' differs")
    key = KEYS[name]
    ref_rows = {tuple(r[k] for k in key): r for r in reference["rows"]}
    cand_rows = {tuple(r[k] for k in key): r for r in candidate["rows"]}
    if ref_rows.keys() != cand_rows.keys():
        problems.append("set of experimental cells differs")
    mismatched = [k for k in ref_rows if k in cand_rows and ref_rows[k] != cand_rows[k]]
    if mismatched:
        problems.append(f"{len(mismatched)} rows differ, e.g. {mismatched[0]}")
    status = "identical" if not problems else "MISMATCH: " + "; ".join(problems)
    print(f"  {label:<10} {name:<12} {len(cand_rows):>4} rows  {status}")
    return not problems


def run_typescript(name, out_dir):
    out = out_dir / f"{name}.json"
    script = ROOT / "src" / f"build_{name}.ts"
    subprocess.run(["node", str(script), "--out", str(out)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-ts", action="store_true", help="skip the TypeScript pipeline")
    args = parser.parse_args()

    from memsig.fast import build_atlas, build_crossdomain

    builders = {"atlas": build_atlas, "crossdomain": build_crossdomain}
    ok = True
    print("Reproduction check against output/*.json")
    with tempfile.TemporaryDirectory() as tmp:
        for name, build in builders.items():
            reference = json.loads((ROOT / "output" / f"{name}.json").read_text(encoding="utf-8"))
            if not args.skip_ts:
                ok &= compare(name, reference, run_typescript(name, Path(tmp)), "TypeScript")
            ok &= compare(name, reference, build(), "Python")
    print("OK — all outputs reproduced exactly." if ok else "FAILED — see mismatches above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
