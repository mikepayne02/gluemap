# Telluride reconstruction contract

This file records the decisions that must survive future sessions. It is a
constraint on the implementation, not a list of experiments.

## Source signals

- Corrected ARKit poses contain reliable capture order, local motion, metric
  scale, and gravity. They contain global drift and a manually corrected reset.
- Polycam LiDAR depth and calibration are metric measurements.
- MapAnything groups jointly predict cameras and geometry. Group membership
  therefore changes the result; it is not per-frame inference.

## Measured pose-conditioning result

The controlled 64-view basement staircase group used the same RGB, metric
depth, confidence, and intrinsics in both branches:

- all corrected poses supplied: staircase separation 1.54--1.56 m, mean LiDAR
  disagreement 1.36 cm;
- no poses supplied: staircase separation 0.40--0.42 m, mean LiDAR
  disagreement 3.68 cm.

Therefore raw poses must not be supplied on both sides of a known drift loop.
This result does not justify disabling pose conditioning everywhere.

## Required behavior

1. Pose-relation confidence filtering may affect global camera averaging only.
   It must never delete an image, LiDAR observation, or MapAnything virtual
   track from the refinement input.
2. Every production camera must have a supported MapAnything relationship.
   Missing rotations or centers are fatal validation errors; production must
   not silently interpolate them from either ARKit or capture order.
3. ARKit gravity may orient the completed model once for export and viewing.
   It is not a per-camera bundle-adjustment residual.
4. Do not add absolute ARKit position residuals to the final optimization.
5. The production cover is pose-free by default. Its recovery policy excludes
   rejected evidence before rebuilding the base cover, then adds balanced
   64-view groups around independently verified revisits. Pose conditioning is
   allowed only inside a configured local recovery group after a controlled
   comparison shows better measured-depth consistency than pose-free input.
   Its relative MapAnything measurements may repair that configured temporal
   interval; ARKit translation is never a global optimization target.
6. Frame 1124 is excluded from every dataset and run.
7. Calibrated intrinsics remain fixed during bundle adjustment.
8. Metric-depth-conditioned groups keep one common scale. Spanning-tree scale
   ratios are initialization artifacts and must not rescale individual groups.
9. Preserve every accepted MapAnything group as one rigid local fragment by
   retaining its anchor-to-member measurements. Shared cameras align the
   overlapping fragments globally. Consecutive-pair consensus is a diagnostic
   and validates coverage; it must not replace each coherent group with an
   independently selected step chain. A configured recovery group may add
   temporal measurements only inside the interval it independently passed.
   ARKit motion is never a global pose constraint.
10. Do not start refinement until the complete 3,119-camera coarse model passes
   measured-LiDAR revisit checks and floor/elevation previews are coherent.
11. Production refinement uses SIFT along the temporal frontend plus
   configuration-verified revisit pairs, together with MapAnything virtual
   tracks (`SV`) at standard unit pixel residuals. Generic nonlocal SIFT
   similarity is not trusted in repetitive rooms. Intrinsics remain fixed;
   ARKit pose, translation, and per-camera gravity residuals remain disabled.
   Both point sets remain in the delivered COLMAP model.

## Current artifacts

- Retained diagnostic baseline (contains the now-rejected 2232/3088 pair):
  `/home/michael/telluride/final_reconstruction`
- Final 136-group hybrid cover:
  `/home/michael/telluride/reconstruction_inputs/fullhouse_groups_final.json`
- Rebuilt 134-group pose-free precursor:
  `/home/michael/telluride/reconstruction_inputs/fullhouse_groups_rebuilt.json`
- Recovery policies are checked in under `configs/`; Telluride frame choices
  are configuration, never solver constants.

An earlier inference checkpoint may seed a standard local partial checkpoint
only for tensor-for-tensor identical group definitions. Cache reuse may be
reordered by exact group membership; changed groups are always inferred again.
The resulting complete checkpoint is ordinary GLUEMAP output; no solved
cameras or geometry are fed back into MapAnything.

The two local exceptions are evidence-backed and configuration-scoped:

- frames 966--978 use a fully local pose-conditioned group because the images
  are nearly featureless wall views. Pose-free groups invented 0.5--0.7 m
  jumps; full local conditioning reduced alternating-frame LiDAR disagreement
  to 2.64 cm median with 97.96% below 10 cm;
- frames 3080--3084 use a pose-free group with depth disabled for manifest
  frames 3081--3083 because RGB is nearly stationary while the saved LiDAR and
  ARKit poses jump. The bad modalities are disabled only inside that group.

The complete cover contains one verified-revisit group, 128 graph-cover
groups, five balanced revisit groups, and two local failure-recovery groups.
Group sizes, rejected evidence, recovery candidates, and their bounded
preferred intervals live in JSON policy, not reconstruction code.

## Accepted final output

The complete S/V camera-refinement trial is not the accepted camera model. Its
reprojection objective converged, but the resulting measured-depth diagnostic
showed visible ghosting and a metric-alignment scale change from 0.978 to
1.267. It also produced extreme sparse-point outliers. Keep that result only as
a solver diagnostic.

The accepted Nerfstudio input keeps the validated 136-group coarse cameras and
calibrated intrinsics fixed, then triangulates the configured SIFT database
against those cameras. This produces useful COLMAP points without allowing
bundle adjustment to deform the coherent camera solve. The accepted artifact
is `/home/michael/telluride/final_reconstruction_v2/accepted_colmap`; its exact
checkpoint, configuration, validations, logs, PLY exports, and dense measured-
LiDAR diagnostics are preserved beside it.
