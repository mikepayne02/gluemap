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
   MapAnything cover. Its optional recovery policy first removes explicitly
   rejected evidence, then adds balanced groups around audited weak
   transitions without hard-coding Telluride frame ranges in Python.
8. `run_polycam_native_gluemap.py` performs MapAnything inference, native
   GLUEMAP assembly, and optional global refinement from the saved inference
   result. Use `--inference-only` when newly recovered groups must pass local
   camera/depth validation before they are allowed into global assembly.

The production contract is:

- MapAnything receives RGB, metric LiDAR depth, and calibrated intrinsics.
- The current Telluride cover has 136 64-view groups: one verified basement
  revisit, 128 graph-cover groups, five balanced revisit groups, and two local
  failure-recovery groups. The base and revisit groups are pose-free. One
  featureless upper-stair interval uses locally validated full pose
  conditioning; one RGB/depth mismatch interval suppresses only its bad depth.
  Group sizes, rejected evidence, recovery candidates, and their bounded
  preferred intervals live in JSON policy, not reconstruction code.
- ARKit is used for reset correction and frontend vicinity proposals only.
- Non-temporal proximity alone is not reconstruction evidence.
- Global assembly preserves every accepted MapAnything group as a rigid local
  fragment at fixed metric scale. Shared cameras align overlapping fragments;
  consecutive-pair consensus remains a coverage and disagreement diagnostic.
  A validated recovery group can contribute extra temporal constraints only
  inside the interval declared in its configuration.
- Global rotations use score-weighted Ceres averaging; the unweighted
  PyCOLMAP rotation pass is not the production solver for this group cover.
- Final refinement uses temporal and configuration-verified SIFT plus
  MapAnything virtual tracks (`SV`) with fixed intrinsics and standard unit
  pixel residuals.
- The runner exposes Ceres' linear solver as `--ba-linear-solver`. Keep
  `auto` for ordinary runs; use `sparse_schur` when a dense overlapping-group
  graph makes iterative Schur fail to produce a finite step. This changes the
  numerical factorization only, not the reconstruction objective.
- `--ba-filter-iterations` controls the existing BA/outlier-filter loop; the
  default remains three. A production run may cap it at two when the third,
  stricter pass removes enough support to make the reduced camera system
  rank-deficient after two already-converged passes.
- Continuation is based on the fraction of 2D observations removed, not a
  dimensionally inconsistent comparison between removed observations and 3D
  point count. A pass below the configured 1% removal threshold terminates
  refinement cleanly.
- Absolute ARKit poses and per-camera gravity are not optimization
  constraints.
- Production preflight fails if any consecutive camera pair lacks a joint
  MapAnything prediction. Global assembly never raises weak scores merely to
  force connectivity and never interpolates unsupported cameras.

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
