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
