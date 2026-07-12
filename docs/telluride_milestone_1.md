# Telluride milestone 1: canonical Polycam audit

The Telluride reconstruction must begin from the untouched Polycam keyframe
export, not from a previously rotated, reset-corrected, or pose-graph-optimized
derivative.

Generate the canonical manifest and calibration report with:

```bash
PYTHONPATH=. python scripts/audit_polycam_dataset.py \
  /path/to/polycam_raw \
  /path/to/dataset_audit
```

This writes:

```text
dataset_audit/manifest.json
dataset_audit/calibration_report.json
```

The manifest records chronological frame identity, all four source paths,
dimensions, RGB- and depth-resolution intrinsics, raw ARKit camera-to-world,
OpenCV-axis camera-to-world, and reset-segment membership. Paths inside each
frame are relative to `source_root`.

## Telluride audit result

The 2026-07-12 audit of the original export found:

- 3,120 complete camera/RGB/depth/confidence associations.
- RGB dimensions: 1024 x 768 for every frame.
- Depth and confidence dimensions: 256 x 192 for every frame.
- Confidence values are exactly 0, 54, and 255.
- Median RGB calibration `(fx, fy, cx, cy)` is
  `(716.3555, 716.3555, 513.08545, 385.18161)`.
- Median camera-center step is 0.09560 m and the 95th percentile is 0.31175 m.
- The sole step above the 2 m audit threshold is 10.53193 m between sequence
  indices 2197 and 2198.
- Rotation determinants remain within approximately 4.6e-7 of one and the
  largest orthogonality error is approximately 6.4e-7.
- Scaling 16-bit depth integers by 0.001 gives metre-scale values consistent
  with the JSON center-depth field at the median to about 1.25 cm. Larger
  outliers mean this field is evidence for units, not a pixel-perfect
  registration test.

## Conventions established in code

- Polycam `t_00..t_23` is treated as ARKit/OpenGL camera-to-world.
- OpenCV camera axes are obtained with
  `c2w_cv = c2w_arkit @ diag(1, -1, -1, 1)`.
- Intrinsics are scaled independently in X and Y when changing resolution;
  this assumes no crop or rotation.
- Projection helpers explicitly implement camera-space Z-depth and have a
  synthetic project/back-project round-trip test.

## Checks still required before reset optimization

The audit deliberately does not claim that these unresolved items are correct:

1. RGB/depth orientation and crop registration need visual overlays across the
   trajectory, especially on depth discontinuities.
2. Z-depth versus range-along-ray must be tested with multi-view RGB-D overlap,
   not inferred solely from the center pixel.
3. Gravity/up must be estimated independently for both reset segments.
4. Full pre- and post-reset diagnostic clouds must be projected from every
   frame before choosing reset correspondences.

The adjacent-frame reset correction in the archived experiments is only an
initialization. It is not an accepted reset transform.

## CPU reset diagnostics

Generate manageable raw segment clouds, the complete camera path, and RGB/depth
overlays locally with:

```bash
PYTHONPATH=. python scripts/project_polycam_diagnostics.py \
  /path/to/dataset_audit/manifest.json \
  /path/to/reset_diagnostics/raw_preview \
  --pixel-step 8 \
  --min-confidence 255 \
  --max-depth-m 6
```

The two `raw_segment_*.ply` files retain RGB colors. The camera-center PLY uses
cyan for segment 0 and magenta for segment 1. Every input frame is processed;
`summary.json` reports how many contributed depth points. Set `--pixel-step 1`
for full-resolution diagnostic clouds after the preview conventions are
accepted.

## Accepted rigid reset initialization

The user manually aligned the two raw segments in CloudCompare while preserving
ARKit +Y gravity and unit scale. The accepted raw-post-world to pre-world rigid
initialization is:

```text
0.990410  0.000000 -0.138157  8.093066
0.000000  1.000000  0.000000  0.821993
0.138157  0.000000  0.990410 -3.931697
0.000000  0.000000  0.000000  1.000000
```

This matrix fixes the coordinate-system discontinuity but intentionally does
not deform either segment. Rotation/gravity and Z alignment are considered more
reliable than X/Y because residual trajectory drift remains.

Generate corrected poses and aligned diagnostics with:

```bash
PYTHONPATH=. python scripts/apply_polycam_reset_transform.py \
  dataset_audit/manifest.json \
  dataset_audit/reset_transform.json \
  dataset_audit/manifest_corrected.json

PYTHONPATH=. python scripts/project_polycam_diagnostics.py \
  dataset_audit/manifest_corrected.json \
  reset_diagnostics/aligned \
  --pose-prefix corrected \
  --label aligned
```

### Residual-drift evidence

Pose-proximity proposals plus visual inspection established:

- Frames around 2232 and 3088 revisit the same basement lower-landing area and
  have camera centers within about 0.08 m after rigid reset alignment.
- Frames 2379-2380 and 3034 unmistakably observe the same basement staircase,
  but their corrected camera centers remain about 1.58-1.60 m apart.
- The staircase mismatch is internal drift within reset segment 1. A single
  reset matrix cannot remove it because both visits receive the same transform.
- A very close pose proposal around frames 790 and 2217 was not a valid visual
  correspondence. This confirms that pose/frustum proximity alone cannot add a
  nonlocal graph edge through walls or repeated structure.

The verified staircase ranges should seed a MapAnything comparison with and
without pose conditioning. Depth and intrinsics remain enabled in both cases.
The predicted local geometry becomes a loop measurement for global
optimization; it must not directly overwrite the corrected ARKit trajectory.
