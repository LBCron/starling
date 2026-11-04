#!/bin/bash
# Kev-0.8B CUDA graph spike, 2026-11-04. Usage: run_all.sh patch|load|placement|bench|load-hog
set -uo pipefail
K=/home/mark/Projects/AI-models/kev
S=$K/spike-2026-11-04
E=$S/export-cuda-fp32
EV=$(cd "$(dirname "$0")" && pwd)
R=$EV/results
PY=$K/env-ortgpu/bin/python
BUNDLE=$HOME/.cache/starlings/models/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
COMMON=(--head "$BUNDLE" --fixtures "$K/fixtures-cpu" --long-fixture "$K/fixtures-long-cpu/documents-v1-dev-318-2k.npz")

spike() { "$PY" -I "$EV/spike_s1.py" "$@"; }

case "$1" in
patch)
  "$PY" -I "$EV/patches/1_failure_a_past_input.py" "$E/model.onnx" "$E/model-p1.onnx" | cut -c1-300
  "$PY" -I "$EV/patches/2_failure_b_mask.py" "$E/model-p1.onnx" "$E/model-p2.onnx"
  "$PY" -I "$EV/patches/3_failure_b_value_info.py" "$E/model-p2.onnx" "$E/model-p3.onnx"
  sha256sum "$E"/model-p*.onnx "$K/onnx-cuda-fp32/model-patched3.onnx"
  ;;
load)
  for g in model model-p1 model-p2 model-p3; do
    spike load --onnx "$E/$g.onnx" --out "$R/load-$g.json"
  done
  spike load --onnx "$K/onnx-cuda-fp16/model.onnx" --out "$R/load-fp16.json"
  spike load --onnx "$BUNDLE/model.onnx" --out "$R/load-current-bundle.json"
  ;;
placement)
  spike placement --onnx "$BUNDLE/model.onnx" "${COMMON[@]}" --out "$R/placement-current.json" 2>"$S/logs/placement-current.stderr"
  spike placement --onnx "$E/model-p3.onnx" "${COMMON[@]}" --out "$R/placement-cuda-fp32-repaired.json" 2>"$S/logs/placement-cuda-fp32-repaired.stderr"
  spike placement --onnx "$K/onnx-cuda-fp16/model.onnx" "${COMMON[@]}" --out "$R/placement-cuda-fp16.json" 2>"$S/logs/placement-cuda-fp16.stderr"
  ;;
bench)
  # One process per cell. $2: result prefix, default "bench".
  prefix=${2:-bench}
  for n in 4 8; do
    for cache in host device; do
      spike bench --onnx "$BUNDLE/model.onnx" "${COMMON[@]}" --label current --cache $cache --questions $n --out "$R/$prefix-current-$cache-q$n.json"
      spike bench --onnx "$E/model-p3.onnx" "${COMMON[@]}" --label cuda-fp32-repaired --cache $cache --questions $n --out "$R/$prefix-cuda-fp32-repaired-$cache-q$n.json"
      spike bench --onnx "$K/onnx-cuda-fp16/model.onnx" "${COMMON[@]}" --label cuda-fp16 --cache $cache --questions $n --out "$R/$prefix-cuda-fp16-$cache-q$n.json"
    done
  done
  ;;
hog)
  # The 8-question cells under N busy CPU loops (default 12, the machine's hardware threads).
  n=${2:-12}
  pids=()
  for i in $(seq "$n"); do
    timeout 300 sh -c 'while :; do :; done' &
    pids+=($!)
  done
  sleep 3
  for cache in host device; do
    spike bench --onnx "$BUNDLE/model.onnx" "${COMMON[@]}" --label current --cache $cache --questions 8 --calls 30 --warmup 5 --out "$R/hog-current-$cache-q8.json"
    spike bench --onnx "$E/model-p3.onnx" "${COMMON[@]}" --label cuda-fp32-repaired --cache $cache --questions 8 --calls 30 --warmup 5 --out "$R/hog-cuda-fp32-repaired-$cache-q8.json"
    spike bench --onnx "$K/onnx-cuda-fp16/model.onnx" "${COMMON[@]}" --label cuda-fp16 --cache $cache --questions 8 --calls 30 --warmup 5 --out "$R/hog-cuda-fp16-$cache-q8.json"
  done
  kill "${pids[@]}"
  wait
  ;;
parity)
  # kev_ort.py fixtures on the repaired graph: 64 records against the PyTorch CPU fixtures.
  "$PY" -I "$EV/../../../model-pipeline/reference/kev/kev_ort.py" fixtures --provider cuda --onnx "$E/model-p3.onnx" --head "$BUNDLE" --fixtures "$K/fixtures-cpu" --out "$R/parity-cuda-fp32-repaired-fixtures.json"
  ;;
*)
  echo "unknown: $1" >&2
  exit 2
  ;;
esac
