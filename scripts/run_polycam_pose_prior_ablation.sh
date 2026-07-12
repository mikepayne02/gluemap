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
    "$output/gluemap_aba_sv_poseprior_030m_3deg" \
    "$output/gluemap_aba_virtual_sv_poseprior_030m_3deg" \
    "$output/diagnostics_sv_poseprior_030m_3deg"
cp "$output/telluride_native_summary.json" \
    "$output/telluride_native_summary.pre_poseprior.json"

cd /workspace/telluride/gluemap
export PYTHONPATH=.:scripts
export LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch

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
    --track-mode SV \
    --pose-prior-position-sigma-m 0.30 \
    --pose-prior-rotation-sigma-deg 3.0 \
    > "$output/refinement_sv_poseprior_030m_3deg.log" 2>&1

test -s "$output/gluemap_aba/images.bin"
test -s "$output/gluemap_aba_virtual/images.bin"
mv "$output/gluemap_aba" "$output/gluemap_aba_sv_poseprior_030m_3deg"
mv "$output/gluemap_aba_virtual" \
    "$output/gluemap_aba_virtual_sv_poseprior_030m_3deg"
cp "$output/telluride_native_summary.json" \
    "$output/telluride_native_summary.sv_poseprior_030m_3deg.json"
python -u /workspace/telluride/inputs/evaluate_polycam_native_reconstruction.py \
    /workspace/telluride/inputs/manifest_corrected.json \
    "$output/telluride_native_summary.sv_poseprior_030m_3deg.json" \
    "$output/gluemap_aba_sv_poseprior_030m_3deg" \
    "$output/diagnostics_sv_poseprior_030m_3deg"

# Restore the pose-prior-free SV control at the conventional path.
cp -a "$output/gluemap_aba_sv" "$output/gluemap_aba"
cp -a "$output/gluemap_aba_virtual_sv" "$output/gluemap_aba_virtual"
cp "$output/telluride_native_summary.pre_poseprior.json" \
    "$output/telluride_native_summary.json"
