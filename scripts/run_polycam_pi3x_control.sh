#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 WAIT_PID OUTPUT_DIRECTORY" >&2
    exit 2
fi

wait_pid="$1"
output="$2"

while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
done

rm -rf "$output"
mkdir -p "$output"

cd /workspace/telluride/gluemap
export PYTHONPATH=.:scripts
export LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u /workspace/telluride/inputs/run_polycam_native_gluemap.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    /workspace/telluride/inputs/native_frontend_edges.csv \
    /workspace/telluride/native_dataset \
    "$output" \
    --source-root /workspace/telluride/data \
    --backend pi3x \
    --manifest-start 1700 \
    --manifest-end 1747 \
    --max-neighbors 25 \
    --num-workers 2 \
    > "$output/inference_and_coarse.log" 2>&1

python -u /workspace/telluride/inputs/evaluate_polycam_native_reconstruction.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    "$output/telluride_native_summary.json" \
    "$output/coarse" \
    "$output/diagnostics_coarse"

python -u /workspace/telluride/inputs/run_polycam_native_gluemap.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    /workspace/telluride/inputs/native_frontend_edges.csv \
    /workspace/telluride/native_dataset \
    "$output" \
    --source-root /workspace/telluride/data \
    --backend pi3x \
    --manifest-start 1700 \
    --manifest-end 1747 \
    --max-neighbors 25 \
    --num-workers 2 \
    --full-refinement \
    --track-mode SV \
    > "$output/refinement_sv.log" 2>&1

python -u /workspace/telluride/inputs/evaluate_polycam_native_reconstruction.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    "$output/telluride_native_summary.json" \
    "$output/gluemap_aba" \
    "$output/diagnostics_refined_sv"
