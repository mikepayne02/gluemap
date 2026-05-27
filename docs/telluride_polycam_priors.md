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
