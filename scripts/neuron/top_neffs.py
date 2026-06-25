"""Print the top-N NEFF hashes by total_time (from *.json summary files).

Used to select the heaviest NEFFs whose NEFF+NTFF pairs are worth preserving
to durable storage for an instruction-level (SQL/parquet) query later — the
summary-json/text aggregates can't name the op, the NTFF trace can.

Usage: python top_neffs.py <json-dir> [N]   (default N=5)
Prints up to N unique hashes, one per line, heaviest first.
"""
import sys
import os
import json
import glob

jdir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

rows = []
for f in glob.glob(os.path.join(jdir, "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    nodes = [v for v in d.values() if isinstance(v, dict)] or [d]
    t = max((nd.get("total_time", 0) or 0) for nd in nodes)
    rows.append((t, os.path.basename(f)[:-5]))

rows.sort(reverse=True)
for _, h in rows[:n]:
    print(h)
