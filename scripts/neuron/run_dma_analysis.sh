#!/bin/bash
# run_dma_analysis.sh — fully automatic instruction-level DMA analysis of ALL
# preserved heavy NEFF+NTFF pairs. No exec, no interaction. Prints to stdout.
#
# Two modes:
#   (a) embedded in a benchmark run — pass the local pairs dir:
#         run_dma_analysis.sh /tmp/results/ntff_pairs
#   (b) standalone — no arg: copy the results tar.gz off the S3 PVC to /tmp,
#       untar, and analyze. Storage rule: /var/mdl is the S3 PVC — ONLY copy the
#       tar.gz off it; all ingest/parquet/queries are LOCAL on /tmp.
#
# neuron-explorer view --output-format parquet WRITES the parquet then idles on
# a UI server (no exit). So we run it backgrounded, wait for the parquet dir size
# to STABILIZE (writing finished), then kill it — that is the completion signal.
set -uo pipefail

WORK=/tmp/ntff_query
TGZ=/var/mdl/wan2-i2v-14b_results.tar.gz
EXPLORER=$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== deps: duckdb (local query engine) ==="
python3 -c "import duckdb" 2>/dev/null || pip install duckdb >/dev/null 2>&1 || { echo "duckdb install FAILED"; exit 1; }

PAIRS="${1:-}"
if [ -n "$PAIRS" ]; then
  echo "=== using provided pairs dir (embedded mode): $PAIRS ==="
  PQROOT="$PAIRS/../dma_pq"
else
  echo "=== standalone: copy results tar.gz off the S3 PVC to local /tmp ==="
  rm -rf "$WORK"; mkdir -p "$WORK"
  cp "$TGZ" "$WORK/results.tar.gz" || { echo "MISSING $TGZ"; exit 1; }
  tar -xzf "$WORK/results.tar.gz" -C "$WORK"
  PAIRS="$WORK/results/ntff_pairs"
  PQROOT="$WORK"
fi
mkdir -p "$PQROOT"
mapfile -t NEFFS < <(ls -S "$PAIRS"/*.neff 2>/dev/null)
echo "found ${#NEFFS[@]} NEFF+NTFF pairs"

ingest() {  # $1=neff $2=ntff $3=outdir — write parquet, then kill the idle server
  local neff="$1" ntff="$2" out="$3"
  rm -rf "$out"; mkdir -p "$out"
  timeout 1200 "$EXPLORER" view -n "$neff" -s "$ntff" \
    --output-format parquet --output-file "$out" \
    --ignore-instruction-hierarchy --ignore-event-trace >/dev/null 2>&1 &
  local pid=$!
  # neuron-explorer creates Summary.parquet early but FILLS IT LAST, then idles
  # on a UI server (never exits). So we wait until the parquet dir size has been
  # STABLE for ~15s (writing finished) — not mere file existence — then kill it.
  local prev=-1 cur stable=0
  for _ in $(seq 1 240); do  # up to 20 min
    kill -0 "$pid" 2>/dev/null || break
    cur=$(du -sb "$out" 2>/dev/null | cut -f1)
    if [ -f "$out/Summary.parquet" ] && [ "$cur" = "$prev" ]; then
      stable=$((stable+1)); [ "$stable" -ge 3 ] && break   # ~15s unchanged
    else
      stable=0
    fi
    prev="$cur"; sleep 5
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # valid only if Summary.parquet is a real file (parquet footer >= a few hundred bytes)
  [ -f "$out/Summary.parquet" ] && [ "$(stat -c%s "$out/Summary.parquet" 2>/dev/null || echo 0)" -gt 1000 ]
}

for neff in "${NEFFS[@]}"; do
  h=$(basename "$neff" .neff)
  ntff="$PAIRS/$h.ntff"
  out="$PQROOT/pq_$h"
  echo ""
  echo "########## ingesting $h ($(du -h "$neff" | cut -f1)) ##########"
  if [ ! -f "$ntff" ]; then echo "  no NTFF, skip"; continue; fi
  if ingest "$neff" "$ntff" "$out"; then
    python3 "$HERE/pq_dma_report.py" "$out" "$h"
    # also dump tensor geometry (read_shape->write_shape, operands) to map the
    # HLO op to a source line — pins WHICH cat/pad/copy this NEFF is.
    [ -f "$HERE/pq_pad_shapes.py" ] && python3 "$HERE/pq_pad_shapes.py" "$out" "$h" 2>/dev/null || true
  else
    echo "  ingest produced no parquet (timeout/too big) — skip"
  fi
done

echo ""
echo "=== DMA analysis complete for all NEFFs ==="
