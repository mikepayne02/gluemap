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
5. The current production group cover is pose-free. ARKit participates in
   reset correction and frontend vicinity proposals, not MapAnything
   conditioning or global optimization.
6. Frame 1124 is excluded from every dataset and run.
7. Calibrated intrinsics remain fixed during bundle adjustment.
8. Metric-depth-conditioned groups keep one common scale. Spanning-tree scale
   ratios are initialization artifacts and must not rescale individual groups.
9. Use score-weighted Ceres rotation averaging. The PyCOLMAP rotation pass
   disconnected 56 otherwise supported cameras in the 130-group production
   cover and is not an accepted production path.
10. Do not start refinement until the complete 3,119-camera coarse model passes
   measured-LiDAR revisit checks and floor/elevation previews are coherent.
11. Production refinement is virtual/neural only (`V`). SIFT is excluded from
   both the camera objective and the delivered COLMAP point cloud.

## Current artifacts

- Active replacement output:
  `/workspace/telluride/results/fullhouse_rebuild`
- Audited 130-group production cover:
  `/workspace/telluride/inputs/fullhouse_groups_rebuild.json`
- Corrected manifest:
  `/workspace/telluride/inputs/manifest_corrected.json`

The previous caches remain diagnostic baselines only. They must not overwrite
or initialize the active replacement result.
