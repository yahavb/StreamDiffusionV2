"""Compact per-NEFF DMA report from a neuron-explorer parquet dump.
Names the op behind the DMA using VERIFIED columns (confirmed against a real
dump on-image). duckdb only, no pandas.

Usage: python3 pq_dma_report.py <parquet-dir> [label]
"""
import sys
import os
import duckdb

pq = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(pq)
S = os.path.join(pq, "Summary.parquet")
I = os.path.join(pq, "Instruction.parquet")
D = os.path.join(pq, "DmaPacketAggregated.parquet")
con = duckdb.connect()

print("\n" + "=" * 76)
print("NEFF:", label)
print("=" * 76)

if not os.path.exists(S) or os.path.getsize(S) < 1000:
    print("  Summary.parquet missing/incomplete — ingest didn't finish, skipping")
    sys.exit(0)

# ── engine / DMA shares (Summary) ──
r = con.execute(f"""
  SELECT total_time, total_active_time_percent,
         tensor_engine_active_time_percent, gpsimd_engine_active_time_percent,
         software_dynamic_dma_active_time_percent, static_dma_active_time_percent,
         hardware_dynamic_dma_active_time_percent, matmul_instruction_count,
         dma_transfer_total_bytes
  FROM read_parquet('{S}')""").fetchone()
tt, act, ten, gps, swd, std, hwd, mm, by = [x if x is not None else 0 for x in r]
print(f"total_time={tt*1e3:7.1f}ms   active={act*100:4.0f}%  (stall={100-act*100:.0f}%)   matmul_count={mm}")
print(f"engine : tensor={ten*100:5.1f}%   gpsimd={gps*100:5.1f}%")
print(f"dma    : sw_dyn={swd*100:5.1f}%   static={std*100:5.1f}%   hw_dyn={hwd*100:5.1f}%   moved={by/1e6:.0f}MB")

# ── top instructions by duration — NAMES the op (hlo + source line) ──
print("\ntop instructions by duration (names the op):")
for eng, op, n, dur, hlo, src in con.execute(f"""
  SELECT engine, opcode, COUNT(*) n, SUM(duration_ns) dur_ns,
         ANY_VALUE(hlo_name) hlo,
         ANY_VALUE(bir_debug_info_source_location) src
  FROM read_parquet('{I}')
  GROUP BY 1,2 ORDER BY dur_ns DESC LIMIT 6""").fetchall():
    print(f"  {eng:7} {op:22} n={n:<7} dur={dur/1e6:7.2f}ms  hlo={(hlo or '')[:60]}  src={(src or '')[-50:]}")

# ── DMA by queue type + direction ──
if os.path.exists(D):
    print("\ndma by queue/direction:")
    for qt, s, d, n, b, dur in con.execute(f"""
      SELECT queue_type, source, dest, COUNT(*) n,
             SUM(transfer_bytes) bytes, SUM(duration_ns) dur_ns
      FROM read_parquet('{D}')
      GROUP BY 1,2,3 ORDER BY dur_ns DESC LIMIT 6""").fetchall():
        print(f"  {qt:18} {s} -> {d}  n={n:<7} {b/1e6:6.0f}MB  dur={dur/1e6:6.2f}ms")
