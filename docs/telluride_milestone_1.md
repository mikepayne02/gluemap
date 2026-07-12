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
