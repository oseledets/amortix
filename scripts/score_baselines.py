"""Score dumped baseline samples against dumped references: the FID step.

Every baseline script (examples/baseline_*.py) dumps per-arm posterior
samples and the reference draws into $AMX_DUMP as ``<sys>_<arm>.npz`` and
``ref_<sys>.npz``.  This script turns those dumps into the median-FID
numbers quoted in the comparison tables, so the whole head-to-head is two
commands: run the baseline script with AMX_DUMP set, then run this.

    python scripts/score_baselines.py <dump_dir> [--out report.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amortix.evaluation import fid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump", help="directory with <sys>_<arm>.npz and ref_<sys>.npz")
ap.add_argument("--out", default=None)
args = ap.parse_args()

refs = {}
for p in glob.glob(os.path.join(args.dump, "ref_*.npz")):
    name = os.path.basename(p)[4:-4]
    refs[name] = np.load(p)["samples"]

report = {}
for p in sorted(glob.glob(os.path.join(args.dump, "*.npz"))):
    base = os.path.basename(p)[:-4]
    if base.startswith("ref_"):
        continue
    sysname, arm = base.split("_", 1)
    if sysname not in refs:
        print(f"{base}: нет референса ref_{sysname}, пропуск")
        continue
    ref = refs[sysname]
    smp = np.load(p)["samples"]
    n = min(len(ref), len(smp))
    vals = np.array([fid(smp[i], ref[i]) for i in range(n)])
    report.setdefault(sysname, {})[arm] = dict(
        n_sets=int(n), fid_median=float(np.median(vals)),
        fid_mean=float(vals.mean()),
        fid_iqr=[float(np.quantile(vals, 0.25)), float(np.quantile(vals, 0.75))])
    print(f"{sysname:6s} {arm:12s} медиана {np.median(vals):8.4f} "
          f"(среднее {vals.mean():.4f}, n={n})")

if args.out:
    json.dump(report, open(args.out, "w"), indent=1)
    print("->", args.out)
