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

test -s "$output/gluemap_aba/images.bin"
rm -rf \
    "$output/gluemap_aba_sv" \
    "$output/gluemap_aba_virtual_sv" \
    "$output/diagnostics_refined_sv" \
    "$output/gluemap_aba_v" \
    "$output/diagnostics_virtual_only"
cp -a "$output/gluemap_aba" "$output/gluemap_aba_sv"
test -s "$output/gluemap_aba_virtual/images.bin"
cp -a "$output/gluemap_aba_virtual" "$output/gluemap_aba_virtual_sv"
cp -a "$output/diagnostics_refined" "$output/diagnostics_refined_sv"
cp "$output/telluride_native_summary.json" \
    "$output/telluride_native_summary.sv.json"

cd /workspace/telluride/gluemap
export PYTHONPATH=.:scripts
export LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch

rm -rf "$output/diagnostics_coarse_calibrated"
python -u /workspace/telluride/inputs/evaluate_polycam_native_reconstruction.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    "$output/telluride_native_summary.sv.json" \
    "$output/coarse" \
    "$output/diagnostics_coarse_calibrated"

python -u /workspace/telluride/inputs/run_polycam_native_gluemap.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    /workspace/telluride/inputs/native_frontend_edges.csv \
    /workspace/telluride/native_dataset \
    "$output" \
    --source-root /workspace/telluride/data \
    --backend map_anything \
    --manifest-start 1500 \
    --manifest-end 2100 \
    --max-neighbors 25 \
    --num-workers 2 \
    --full-refinement \
    --track-mode V \
    > "$output/refinement_virtual_only.log" 2>&1

test -s "$output/gluemap_aba/images.bin"
mv "$output/gluemap_aba" "$output/gluemap_aba_v"
cp "$output/telluride_native_summary.json" \
    "$output/telluride_native_summary.v.json"
python -u /workspace/telluride/inputs/evaluate_polycam_native_reconstruction.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    "$output/telluride_native_summary.v.json" \
    "$output/gluemap_aba_v" \
    "$output/diagnostics_virtual_only"

# Leave the SIFT+virtual result at the conventional path for downstream tools.
cp -a "$output/gluemap_aba_sv" "$output/gluemap_aba"
cp "$output/telluride_native_summary.sv.json" \
    "$output/telluride_native_summary.json"
