"""Pick which NEFFs to ingest into neuron-explorer, by TWO criteria so neither
a fat-file nor a long-running graph is missed:
  - largest total_time (from the *.json summary files in <json-dir>)
  - largest file size  (from the *.neff files in <neff-dir>)
Prints up to two UNIQUE hashes, one per line (dedup if both agree).
Usage: python pick_neffs.py <json-dir> <neff-dir>"""
import sys
import os
import json
import glob

jdir, ndir = sys.argv[1], sys.argv[2]

# by total_time
best_t, bt = None, -1.0
for f in glob.glob(os.path.join(jdir, "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    nodes = [v for v in d.values() if isinstance(v, dict)] or [d]
    t = max((n.get("total_time", 0) or 0) for n in nodes)
    if t > bt:
        bt, best_t = t, os.path.basename(f)[:-5]

# by file size
best_s, bs = None, -1
for f in glob.glob(os.path.join(ndir, "*.neff")):
    try:
        sz = os.path.getsize(f)
    except OSError:
        continue
    if sz > bs:
        bs, best_s = sz, os.path.basename(f)[:-5]

seen = []
for h in (best_t, best_s):
    if h and h not in seen:
        seen.append(h)
        print(h)
