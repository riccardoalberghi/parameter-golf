#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_config_tests.sh — sweep new Monarch configs, 10 min each, then report.
#
# Usage:
#   bash run_config_tests.sh              # auto-detect GPUs
#   NGPUS=1 bash run_config_tests.sh      # force 1 GPU
#   QUICK=1 bash run_config_tests.sh      # 2 min smoke runs
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"
WALLCLOCK="${QUICK:+120}"          # 2 min if QUICK=1, else default 600
WALLCLOCK="${WALLCLOCK:-600}"
LOGDIR="config_test_logs"
REPORT="$LOGDIR/report.md"
mkdir -p "$LOGDIR"

# Common env shared by every run
COMMON=(
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  MAX_WALLCLOCK_SECONDS="$WALLCLOCK"
  VAL_LOSS_EVERY=1000
  TRAIN_LOG_EVERY=200
)

# ── Test matrix ──────────────────────────────────────────────────────────────
# Each entry: NAME  extra env vars (space-separated KEY=VALUE)
declare -a CONFIGS=(
  # 1. Monarch on attention (Q,K,V,O) — muon optim
  "monarch_attn_muon"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_OPTIM=muon"

  # 2. Monarch on MLP only — muon optim
  "monarch_mlp_muon"
  "MONARCH_FACTORS=2 MONARCH_MLP=1 MONARCH_OPTIM=muon"

  # 3. Monarch on everything — muon optim
  "monarch_all_muon"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=muon"

  # 4. Monarch on everything except K — muon optim
  "monarch_all_no_k_muon"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=muon"

  # 5. Monarch on everything — block_svd optim
  "monarch_all_block_svd"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=block_svd"

  # 6. Monarch on everything — block_ns optim
  "monarch_all_block_ns"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=block_ns"

  # 7. Monarch on everything — muon + Givens quant
  "monarch_all_muon_givens"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=muon MONARCH_GIVENS_QUANT=1"

  # 8. Upstream baseline — no monarch
  "baseline"
  ""

  # 9. Monarch on everything — block_svd + Givens quant
  "monarch_all_block_svd_givens"
  "MONARCH_FACTORS=2 MONARCH_Q=1 MONARCH_K=1 MONARCH_V=1 MONARCH_O=1 MONARCH_MLP=1 MONARCH_OPTIM=block_svd MONARCH_GIVENS_QUANT=1"
)

NUM_CONFIGS=$(( ${#CONFIGS[@]} / 2 ))
echo "=== Config sweep: $NUM_CONFIGS runs × ${WALLCLOCK}s each on $NGPUS GPU(s) ==="
echo ""

# ── Run each config ──────────────────────────────────────────────────────────
for (( idx=0; idx < ${#CONFIGS[@]}; idx+=2 )); do
  NAME="${CONFIGS[$idx]}"
  EXTRA="${CONFIGS[$idx+1]}"
  LOGFILE="$LOGDIR/${NAME}.log"
  RUN_NUM=$(( idx / 2 + 1 ))

  echo "[$RUN_NUM/$NUM_CONFIGS] Running: $NAME"
  echo "  Extra env: ${EXTRA:-<none>}"
  echo "  Log: $LOGFILE"

  # Build env array
  ENV=("${COMMON[@]}" RUN_ID="$NAME" SEED=1337)
  for kv in $EXTRA; do
    ENV+=("$kv")
  done

  # Run training, tee to log
  env "${ENV[@]}" torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py \
    > "$LOGFILE" 2>&1 || {
      echo "  !! FAILED (exit $?) — see $LOGFILE"
      continue
    }

  echo "  Done."
  echo ""
done

# ── Generate report ──────────────────────────────────────────────────────────
echo "=== Generating report ==="

{
  echo "# Config Sweep Report"
  echo ""
  echo "Wallclock: ${WALLCLOCK}s per run | GPUs: $NGPUS | $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo ""
  echo "| # | Config | val\_bpb (quant) | val\_bpb (pre-quant) | Steps | Step avg (ms) | Artifact (bytes) | .ptz (bytes) |"
  echo "|---|--------|----------------:|--------------------:|------:|--------------:|-----------------:|-------------:|"

  run=0
  for (( idx=0; idx < ${#CONFIGS[@]}; idx+=2 )); do
    NAME="${CONFIGS[$idx]}"
    LOGFILE="$LOGDIR/${NAME}.log"
    run=$((run + 1))

    if [ ! -f "$LOGFILE" ]; then
      echo "| $run | $NAME | MISSING | — | — | — | — | — |"
      continue
    fi

    # Final quantized val_bpb (the number that matters for the competition)
    QBPB=$(grep 'final_int8_zlib_roundtrip_exact' "$LOGFILE" | tail -1 \
           | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")

    # Last pre-quantization val_bpb
    PBPB=$(grep -P '^step:\d+/\d+ val_loss' "$LOGFILE" | tail -1 \
           | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")

    # Last step number achieved
    STEPS=$(grep -P '^step:\d+/\d+ train_loss' "$LOGFILE" | tail -1 \
            | grep -oP 'step:\K\d+' || echo "—")

    # Step average time from last train log line
    STEPAVG=$(grep -P '^step:\d+/\d+ train_loss' "$LOGFILE" | tail -1 \
              | grep -oP 'step_avg:\K[0-9.]+' || echo "—")

    # Artifact = total submission size (model + code), the 16MB budget number
    ARTIFACT=$(grep 'Total submission size int8' "$LOGFILE" | tail -1 \
               | grep -oP ':\s*\K[0-9]+' || echo "—")

    # Compressed model .ptz alone
    PTZ=$(grep 'Serialized model int8+zlib' "$LOGFILE" | tail -1 \
          | grep -oP 'int8\+zlib:\s*\K[0-9]+' || echo "—")

    echo "| $run | $NAME | $QBPB | $PBPB | $STEPS | $STEPAVG | $ARTIFACT | $PTZ |"
  done

  echo ""
  echo "---"
  echo ""

  # Per-config detail: last 5 train log lines + final lines
  for (( idx=0; idx < ${#CONFIGS[@]}; idx+=2 )); do
    NAME="${CONFIGS[$idx]}"
    LOGFILE="$LOGDIR/${NAME}.log"
    [ -f "$LOGFILE" ] || continue

    echo "### $NAME"
    echo ""
    echo '```'
    grep -P '^step:\d+/\d+ train_loss' "$LOGFILE" | tail -5
    grep 'final_int8_zlib_roundtrip' "$LOGFILE"
    grep 'Total submission size' "$LOGFILE" | tail -1
    echo '```'
    echo ""
  done

} > "$REPORT"

echo ""
echo "=== Report written to $REPORT ==="
echo ""
cat "$REPORT"
