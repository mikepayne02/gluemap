# Telluride scripts

This directory contains one production reconstruction path plus read-only
diagnostics. Do not create parallel reconstruction pipelines here. Extend the
canonical stages below only when the existing stage cannot express a required
operation.

## Production path

Run these stages in order:

1. `audit_polycam_dataset.py` creates the canonical manifest.
2. `apply_polycam_reset_transform.py` applies the single accepted reset
   transform between the two ARKit tracking segments.
3. `prepare_polycam_native_dataset.py` creates the upright, ordered image set
   and excludes the thumb frame.
4. `analyze_polycam_pose_candidates.py` proposes nearby non-temporal views.
5. `audit_polycam_visual_edges.py` measures independent image support for
   proposed non-temporal edges.
6. `build_polycam_native_frontend.py` freezes the audited temporal and
   image-verified edge list.
7. `build_polycam_native_groups.py` creates the overlapping 64-view
   MapAnything cover, including verified revisit groups.
8. `run_polycam_native_gluemap.py` performs MapAnything inference, native
   GLUEMAP assembly, and optional global refinement from the saved inference
   result.

The production contract is:

- MapAnything receives RGB, metric LiDAR depth, and calibrated intrinsics.
- ARKit is used for reset correction and frontend vicinity proposals only.
- Non-temporal proximity alone is not reconstruction evidence.
- Global assembly uses overlapping MapAnything group estimates at fixed
  metric scale.
- Global rotations use score-weighted Ceres averaging; the unweighted
  PyCOLMAP rotation pass is not the production solver for this group cover.
- Final refinement uses virtual/neural tracks (`V`) with fixed intrinsics.
- SIFT, absolute ARKit poses, and per-camera gravity are not optimization
  constraints.

## Acceptance and exports

- `validate_polycam_bridge_geometry.py` is the required geometry gate. It
  compares measured-LiDAR surfaces from independently scanned visits using the
  solved cameras and reports the known staircase closure distances.
- `export_polycam_solved_depth_cloud.py` back-projects measured LiDAR through
  a solved COLMAP trajectory. It does not alter the reconstruction.
- `render_polycam_dense_diagnostics.py` renders top-down floor slices and
  orthographic sections from that dense diagnostic cloud.

## Diagnostic-only tools

These scripts never supply optimization targets:

- `evaluate_polycam_native_reconstruction.py` reports post-hoc disagreement
  with ARKit and exports a sparse diagnostic.
- `project_polycam_diagnostics.py` visualizes the audited input/reset state.
- `render_polycam_floor_slices.py` is the older PLY renderer; prefer
  `render_polycam_dense_diagnostics.py` for final validation.

Do not add a script merely to encode another parameter combination. The runner
and group configuration are the supported variation points. Remove a superseded
script instead of leaving two authoritative implementations.
