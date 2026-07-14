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
2. If a camera lacks a supported MapAnything pose relationship, initialize it
   from the aligned corrected ARKit trajectory. Never replace a missing run by
   a straight line when the measured trajectory exists.
3. Use ARKit gravity during coarse assembly and bundle adjustment. Gravity
   constraints must not constrain yaw or translation.
4. Do not add absolute ARKit position residuals to the final optimization.
5. Pose conditioning is group-local and explicit. The default selective mode
   supplies at most eight corrected poses near the group anchor. Distant
   revisit members remain unposed so MapAnything can correct accumulated
   drift. Known unreliable anchor ranges remain entirely pose-free.
6. Frame 1124 is excluded from every dataset and run.
7. Calibrated intrinsics remain fixed during bundle adjustment.
8. Do not start a multi-hour refinement until the complete 3,119-camera coarse
   model and measured-LiDAR floor/elevation previews are coherent.

## Current artifacts

- Pose-free full-house MapAnything cache:
  `/workspace/telluride/results/fullhouse_depth64/star_result.pth`
- Original 134-group cover:
  `/workspace/telluride/inputs/fullhouse_groups_mapanything_depth64.json`
- Selective-pose version of the same cover:
  `/workspace/telluride/inputs/fullhouse_groups_mapanything_depth64_sparse_pose.json`
- Corrected manifest:
  `/workspace/telluride/inputs/manifest_corrected.json`

The cached pose-free inference remains useful. Selective pose inference must be
written to a distinct final result directory until its complete coarse model is
validated; it must not partially overwrite the pose-free cache.
