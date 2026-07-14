# Telluride RunPod Environment

This file records the working native GLUEMAP + MapAnything environment used for the Telluride reconstruction. The RunPod network volume persists, but packages installed into the container's system Python do not survive a stopped/restarted pod.

## Pod image and runtime

- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Python: 3.11
- PyTorch: `2.4.1+cu124`
- Torchvision: `0.19.1+cu124`
- GPU used successfully: NVIDIA A40 48 GB
- Network volume: `/workspace`
- Repository: `/workspace/telluride/gluemap`
- Dataset: `/workspace/telluride/data`
- Hugging Face cache: `/workspace/cache/huggingface`
- Torch cache: `/workspace/cache/torch`

Use these cache variables for model runs:

```bash
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Native GLUEMAP C++ dependencies

Ubuntu 22.04 provides Ceres 2.0, which is too old for GLUEMAP's `ceres::Manifold` API. Do not build against `/usr/lib` Ceres.

Micromamba is stored persistently at:

```text
/workspace/tools/micromamba-bin/micromamba
```

The persistent C++ dependency prefix is:

```text
/workspace/tools/gluemap-cpp
```

It was created with:

```bash
/workspace/tools/micromamba-bin/micromamba create -y \
  -p /workspace/tools/gluemap-cpp -c conda-forge \
  ceres-solver=2.2.0 eigen=3.4.0 metis=5.1.0 \
  boost=1.85.0 libstdcxx-ng=15.2.0
```

System build/runtime packages:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cmake build-essential libeigen3-dev libceres-dev libmetis-dev \
  libboost-dev pybind11-dev libsm6 libice6
```

Build the native extension against Ceres 2.2:

```bash
cd /workspace/telluride/gluemap
CMAKE_PREFIX_PATH=/workspace/tools/gluemap-cpp \
CMAKE_BUILD_PARALLEL_LEVEL=9 \
LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib \
python -m pip install -e . --no-deps
```

The runtime must expose the same C++ prefix:

```bash
export LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib:${LD_LIBRARY_PATH:-}
```

Verify native imports together with Torch:

```bash
LD_LIBRARY_PATH=/workspace/tools/gluemap-cpp/lib python - <<'PY'
import torch
import pygluemap
import pycolmap
import pyceres
print(torch.__version__, torch.cuda.is_available())
print(pygluemap.__file__, pycolmap.__version__, pyceres.__version__)
PY
```

## Python package warning

Install `pycolmap-cuda12` without dependency resolution:

```bash
python -m pip install --no-deps pycolmap-cuda12==4.0.4
python -m pip install --no-deps pyceres==2.6 faiss-cpu==1.13.2
```

Do **not** run an unconstrained `pip install pycolmap-cuda12`. Its `cuda-toolkit` dependency can make pip replace the working PyTorch/CUDA stack and overlay CUDA 11 NCCL files. If Torch is damaged, repair the complete CUDA 12.4 dependency set with:

```bash
python -m pip install --force-reinstall \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

Then reinstall `pycolmap-cuda12==4.0.4` with `--no-deps`.

## Model packages

The vendored model submodules are under `thirdparty/`. After a fresh container start:

```bash
cd /workspace/telluride/gluemap
python -m pip install -e thirdparty/mapanything
python -m pip install -e thirdparty/pi3 --no-deps
python -m pip install \
  'git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8'
```

MapAnything 128-view RGB + metric-depth + intrinsics inference completed on the A40 with 18.31 GB peak allocated and 20.99 GB reserved at 518x392 output resolution.

## Restart checklist

1. Leave the pod running unless the user explicitly requests a stop.
2. Confirm `/workspace/telluride`, `/workspace/cache`, and `/workspace/tools/gluemap-cpp` are mounted.
3. Restore the pinned PyTorch 2.4.1/CUDA 12.4 runtime if necessary.
4. Install model packages and Python-only GLUEMAP dependencies.
5. Rebuild/install the editable native extension against `/workspace/tools/gluemap-cpp`.
6. Export `LD_LIBRARY_PATH`, `HF_HOME`, and `TORCH_HOME` as above.
7. Run the combined import verification before inference or global mapping.

## Preserved reconstruction data

- 64-view pose-free predictions: `/workspace/telluride/results/round2_posefree`
- Restored rigid baseline: `/workspace/telluride/results/round2_rigid`
- 128-view MapAnything stair diagnostic: `/workspace/telluride/results/stair_bridge_128`
- Frontend LightGlue audits: `/workspace/telluride/results/frontend_audit`

The rejected per-camera translation optimizer and its point clouds were removed. They must not be used as initialization for native GLUEMAP.

## Telluride-native reconstruction conventions

- Polycam ARKit poses select nearby candidate frames and remain an external
  metric/global diagnostic; they are not currently fed into MapAnything.
- The audited frontend graph combines temporal continuity, reciprocal LiDAR
  overlap, and LightGlue verification. Its frozen edge list is
  `/workspace/telluride/inputs/native_frontend_edges.csv`.
- Frame 1124 contains the operator's thumb and is excluded from the upright
  native dataset and all reconstruction runs.
- MapAnything receives upright RGB, metric LiDAR depth, and calibrated
  intrinsics. The native dataset is `/workspace/telluride/native_dataset`.
- Native global mapping replaces MapAnything's predicted intrinsics with the
  exact upright Polycam calibration before writing COLMAP, and native bundle
  adjustment keeps those calibration parameter blocks fixed. Camera motion
  must not be hidden as a per-frame focal-length change.
- `use_dummy_tracks=True` avoids loading VGGSfM. Those repeated query points
  are interface placeholders, not real cross-view tracks. Production uses
  `track_mode=V`, so GLUEMAP skips SIFT construction and track snapping and
  refines only MapAnything depth-derived virtual tracks.
- Virtual-only BA must explicitly install quaternion manifolds, fix one camera
  pose, fix one translation component on a well-separated second camera, and
  keep calibrated intrinsics constant. PyCOLMAP cannot establish that gauge
  when its real/SIFT reconstruction has zero tracks. Results produced with the
  `Failed to fix Gauge` warning before this setup are invalid controls.
- Corrected ARKit poses are a frontend vicinity/reset signal and a diagnostic,
  not ground truth. The normal `SV` and `V` optimization paths do not add an
  ARKit residual. Do not pass `--pose-prior-position-sigma-m` in Telluride
  production runs; differences after diagnostic Sim3 alignment must be called
  ARKit disagreement, not camera error.
- Production global assembly requires every camera to retain MapAnything group
  support. ARKit and temporal interpolation fallbacks are disabled; an
  unsupported camera fails the run instead of receiving an invented pose.
- Metric-depth-conditioned groups all use scale 1.0 during similarity
  averaging. Do not freeze noisy spanning-tree scale ratios as per-group
  scales.
- A validated 48-frame native run is preserved at
  `/workspace/telluride/results/native_integration_1700_1747`. After one
  diagnostic Sim3 to ARKit its camera-center errors were 2.08 cm median,
  3.70 cm p95, and 4.72 cm maximum.
- The active 601-frame closet-region run is
  `/workspace/telluride/results/native_closet_1500_2100`. It may be resumed
  from `star_result.pth`; full refinement does not require rerunning neural
  inference.
- Native star inference writes an atomic `star_result.pth.partial` every 100
  stars. A restarted native runner skips completed star indices and removes
  the partial only after the final `star_result.pth` is safely written. Resume
  is intentionally limited to the single-process path used on the A40.

Do not reintroduce the rejected per-camera translation optimizer, selective
camera anchoring, or independent per-star transforms. Those experiments
sheared otherwise rigid local reconstructions and were materially worse than
the reset-corrected ARKit/LiDAR baseline.

## Observed package versions on the working pod

The editable repository metadata is the intended dependency baseline, but the
currently working container reports a few newer transitive packages:

```text
numpy==2.4.4
opencv-python==5.0.0.93
opencv-python-headless==4.10.0.84
safetensors==0.8.0
scipy==1.17.1
torch==2.4.1+cu124
torchvision==0.19.1+cu124
pycolmap-cuda12==4.0.4
pyceres==2.6
```

Do not normalize or downgrade those packages on a working pod merely to match
the pins. For a fresh pod, start from the pinned repository environment and
run the combined import check before changing any version.

The development test runner was installed separately with `pip install
pytest`; it is optional for inference. After changing `gluemap/pybind`, rebuild
the editable package against `/workspace/tools/gluemap-cpp` before launching a
new process. A running Python process continues using the native module it
already loaded.
