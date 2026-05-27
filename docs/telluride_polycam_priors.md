# Telluride Polycam Priors

This branch adds an opt-in MapAnything path for Polycam RGB-D priors.

## Source Of Truth

Use only raw Polycam-derived data as the prior source:

- RGB: `polycam_raw_adjacent/images`
- Depth: `polycam_raw_adjacent/depth_up`
- Confidence mask: `polycam_raw_adjacent/masks_midhigh`
- Reset-aware poses/intrinsics: `polycam_raw_adjacent/transforms.json`

`polycam_raw_adjacent/transforms.json` is generated outside GlueMap from raw
Polycam camera JSON and only corrects the hard ARKit tracking reset by chaining
the adjacent relative transform across the reset. It is not a prior
reconstruction from DA3, VGGT, MP-SfM, Blender, or AGS.

## What The Patch Does

When the following flags are enabled:

```yaml
polycam_priors_path: /workspace/polycam_raw_adjacent/transforms.json
mapanything_use_polycam_depth: true
mapanything_use_polycam_intrinsics: true
mapanything_use_polycam_pose: false
```

GlueMap loads the Polycam depth, mask, intrinsics, and optional poses for each
star batch. The priors are resampled with the same resize/pad transform used
for the RGB tensor before they are passed to MapAnything.

The MapAnything adapter passes these fields into `model.infer()` through
per-view dictionaries:

- `depth_z`
- `intrinsics`
- `camera_poses` when explicitly enabled
- `is_metric_scale`

Depth and intrinsics are safer first-pass priors. Pose priors should be tested
separately because they can bias MapAnything toward the reset-corrected ARKit
trajectory.

## First Run Config

Use:

```bash
python demo.py --config configs/telluride_polycam_mapanything.yaml
```

The initial config uses:

- `chosen_model: map_anything`
- `intrinsics_mode: PER_FOLDER`
- sequential retrieval enabled
- Polycam depth and intrinsics enabled
- Polycam pose disabled for the first real baseline

For the current flat image folder, `PER_FOLDER` behaves like `SHARED`. If the
dataset is later split into physical subfolders, `PER_FOLDER` avoids mixing
intrinsic buckets across folders.

## 2026-05-27 GlueMap Smoke Findings

The first MapAnything adapter emitted absolute inverse poses for each star:

```python
extrinsics = inv(camera_poses)
```

That was wrong for GlueMap. The Pi3 adapter emits first-view-relative star
extrinsics:

```python
extrinsics = inv(camera_poses) @ camera_poses[:, :1]
```

The MapAnything adapter now matches that convention.

Step-40 smoke diagnostics:

- Depth/intrinsics-only after the fix is still not usable. The graph relaxes
  Doppelgangers filtering to `0.50`, filters most edges, and free similarity
  averaging is unstable/collapsed. This points at the star graph being too weak
  or ambiguous for this indoor/staircase sequence.
- Pose-prior mode after the fix converges cleanly, but it follows
  `polycam_raw_adjacent/transforms.json` almost exactly. The exported coarse
  camera centers differ from the Polycam preseed by only ~3.4 cm median and
  ~8.5 cm p90 after alignment.
- Therefore, pose-prior mode is not currently correcting the basement/entry
  errors. It is mostly preserving the adjacent-reset Polycam trajectory.

Conclusion: the raw RGB/depth/intrinsics look sane, but
`polycam_raw_adjacent/transforms.json` is not a golden trajectory. It only
fixes the hard reset with an adjacent transform and should not be treated as a
truth source. GlueMap needs a better graph/constraint strategy before scaling:
sequential and spatial edges from Polycam, vetted reset/stair loop edges, and
avoidance of weak retrieval edges that only survive after threshold relaxation.

## Next Reconstruction Plan

Use GlueMap as the global geometry/BA engine, but stop letting
SALAD/Doppelgangers choose the graph for this repeated-staircase indoor scan.

1. Generate a deterministic pair graph from Polycam timing and camera centers:
   dense sequential edges, nearby-in-space edges, and explicit vetted loop
   edges around stair/reset regions.
2. Run GlueMap with `--pair_graph_path ... --skip_doppelgangers`, so every
   edge in the graph is intentional and no edge is admitted only because DG
   relaxed to a weak threshold.
3. Feed MapAnything depth, intrinsics, and pose priors for the real run. The
   no-pose branch collapsed on this sequence; pose priors keep the graph
   metric while VGGSfM/GlueMap refinement supplies the correction signal.
4. For coarse diagnostics use `--coarse_only --use_dummy_tracks` to inspect
   cameras quickly.
5. For the real run, remove `--coarse_only` and do not pass
   `--use_dummy_tracks`; this lets VGGSfM run alongside MapAnything. VGGSfM
   should help textured areas snap into place, while MapAnything's virtual
   depth tracks carry low-texture walls.

Implementation notes:

- `scripts/build_polycam_pair_graph.py` writes the graph JSON.
- `--pair_graph_path` bypasses SALAD descriptor loading in sequential mode.
- `--skip_doppelgangers` skips DG scoring and assigns all supplied graph pairs
  score `1.0`.
- `configs/telluride_polycam_mapanything_pairgraph.yaml` is the intended
  starting config for this branch.

## Step-20 Pair Graph Result

The first useful full-refinement run used:

```text
pair_graph_step20_seq8_spatial.json
sample_frequency=20
sequential_window=8
spatial_radius=1.25
spatial_max_neighbors=8
skip_doppelgangers=true
mapanything_use_polycam_depth=true
mapanything_use_polycam_intrinsics=true
mapanything_use_polycam_pose=true
VGGSfM tracks enabled
```

Output:

```text
/workspace/results/gluemap_telluride_step20_pairgraph_seq8_spatial_pose_vggsfm_full/gluemap_aba
```

Logs showed:

```text
156 registered images
14708 sparse points
0 inconsistent rotation edges
0 invalid relative-rotation pairs
Similarity averaging converged
Augmented BA converged
Total pipeline time: 5627 s
```

After Sim3 fitting the optimized keyframe centers back to the reset-corrected
Polycam frame:

```text
scale = 0.998581
mean residual = 0.0308 m
median residual = 0.0282 m
p95 residual = 0.0595 m
max residual = 0.0986 m
```

This run did not introduce a global scale collapse. Remaining quality issues
should be evaluated as local geometry/point consistency problems, especially
around entry/stair/basement overlap, rather than a gross trajectory failure.

Remote setup notes:

```text
Base image used successfully:
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

Repo:
/workspace/repos/gluemap

Env:
/workspace/venvs/gluemap_mm

VGGSfM checkpoint:
/workspace/repos/gluemap/checkpoints/vggsfm_v2_0_0_track_predictor.bin

Kornia pin:
pip install "kornia==0.6.12"

Runtime PYTHONPATH:
/workspace/repos/gluemap/thirdparty/mapanything
```
