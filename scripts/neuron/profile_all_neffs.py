"""Aggregate per-NEFF neuron-profile summary-json files into a model-level
engine breakdown. Answers "where does the time ACTUALLY go" across ALL NEFFs,
not one blended/partial capture.

Each NEFF is weighted by total_time * exec_count (exec counts passed via a
{hash: count} json) so a fast NEFF run 32x outranks a slow one run once.

Usage: python profile_all_neffs.py <dir-of-*.json> [exec_counts.json]

Verbatim port of pave-unrolling/scripts/neuron/profile_all_neffs.py — the
neuron-profile summary-json schema is model-agnostic, so the EBME breakdown is
directly comparable to UniVR's GpSimd-46%/Tensor-0.8% table.
"""
import sys
import os
import json
import glob

ENGINES = [
    ("gpsimd", "gpsimd_engine_active_time"),
    ("tensor", "tensor_engine_active_time"),
    ("vector", "vector_engine_active_time"),
    ("scalar", "scalar_engine_active_time"),
    ("sync", "sync_engine_active_time"),
    ("dma", "dma_active_time"),
    ("sw_dyn_dma", "software_dynamic_dma_active_time"),
    ("hw_dyn_dma", "hardware_dynamic_dma_active_time"),
    ("static_dma", "static_dma_active_time"),
]


def pick_node(d):
    nodes = [v for v in d.values() if isinstance(v, dict)] if isinstance(d, dict) else d
    if not nodes:
        nodes = [d]
    return max(nodes, key=lambda x: x.get("total_time", 0) or 0)


def main():
    jdir = sys.argv[1]
    counts = {}
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        counts = json.load(open(sys.argv[2]))

    files = sorted(glob.glob(os.path.join(jdir, "*.json")))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n = pick_node(d)
        h = os.path.basename(f).replace(".json", "")
        g = lambda k: float(n.get(k, 0) or 0)
        cnt = int(counts.get(h, 1))
        rows.append((h, cnt, n, g))

    if not rows:
        print("NO valid per-NEFF json found — capture failed.")
        return

    print(f"=== {len(rows)} NEFFs profiled ===\n")
    rows.sort(key=lambda r: r[3]("total_time") * r[1], reverse=True)
    print(f"{'neff':14} {'cnt':>4} {'tot_us':>9} {'wtot_us':>10} {'gpsimd%':>8} {'tensor%':>8} {'vector%':>8} {'dma%':>7}")
    agg = {k: 0.0 for k, _ in ENGINES}
    wtot_all = 0.0
    for h, cnt, n, g in rows:
        tot = g("total_time") * 1e6
        wtot = tot * cnt
        wtot_all += wtot
        for k, key in ENGINES:
            agg[k] += g(key) * cnt
        pct = lambda key: g(key + "_percent") * 100
        print(f"{h[:14]:14} {cnt:>4} {tot:>9.1f} {wtot:>10.1f} "
              f"{pct('gpsimd_engine_active_time'):>8.1f} {pct('tensor_engine_active_time'):>8.1f} "
              f"{pct('vector_engine_active_time'):>8.1f} {pct('dma_active_time'):>7.1f}")

    print(f"\n=== MODEL-LEVEL engine time (sum of active_time*exec_count) ===")
    total_engine = sum(agg.values()) or 1.0
    for k, _ in ENGINES:
        print(f"  {k:12} = {agg[k]*1e6:>12.1f} us  ({100*agg[k]/total_engine:>5.1f}% of summed engine time)")
    print(f"\n  weighted total_time across all NEFF executions = {wtot_all:.1f} us")


if __name__ == "__main__":
    main()
