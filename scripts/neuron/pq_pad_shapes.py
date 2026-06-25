"""For a parquet dump of a pad/copy NEFF, print the tensor SHAPES being moved,
so we can identify WHICH source-level op it is (K/V attention pad vs latent
seq_len pad vs context pad, etc.). Shapes come from verified columns. duckdb only.

Usage: python3 pq_pad_shapes.py <parquet-dir> [label]
"""
import sys
import os
import duckdb

pq = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(pq)
I = os.path.join(pq, "Instruction.parquet")
D = os.path.join(pq, "DmaPacketAggregated.parquet")
con = duckdb.connect()

print("\n===== SHAPES for %s =====" % label)

# The HLO op + its operands/attrs (operands often carry shapes/dims)
print("\n-- pad/concat instruction operands & hlo_attrs --")
for hlo, op, operands, attrs, n in con.execute(f"""
  SELECT hlo_name, opcode, ANY_VALUE(operands), ANY_VALUE(hlo_attrs), COUNT(*) n
  FROM read_parquet('{I}')
  WHERE hlo_name LIKE '%pad%' OR hlo_name LIKE '%concat%'
  GROUP BY 1,2 ORDER BY n DESC LIMIT 10""").fetchall():
    print(f"  {hlo}  op={op}  n={n}")
    print(f"    operands: {operands}")
    print(f"    attrs   : {attrs}")

# DMA read/write shapes — the actual tensor geometry moved
print("\n-- DMA read_shape -> write_shape (the tensor being moved) --")
for rs, ws, src, dst, n, by in con.execute(f"""
  SELECT read_shape, write_shape, source, dest, COUNT(*) n, SUM(transfer_bytes) bytes
  FROM read_parquet('{D}')
  GROUP BY 1,2,3,4 ORDER BY bytes DESC LIMIT 10""").fetchall():
    print(f"  {src} {rs} -> {dst} {ws}   n={n}  {by/1e6:.0f}MB")
