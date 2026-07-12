# Telluride conditioned MapAnything validation gate

Do not begin a full GLUEMAP reconstruction until this local-group comparison
passes. The first group contains two visually verified observations of the same
basement staircase whose corrected ARKit camera centers remain about 1.6 m
apart.

## Group

`configs/telluride_mapanything_groups.json` defines a 64-view
`basement_stair_bridge` group:

```text
frames 2369-2400
frames 3009-3040
```

Frames 2379/2380 and 3034 visibly observe the same staircase. This is a real
loop, not a pose-proximity guess.

All inputs are rotated 90 degrees clockwise to portrait/upright orientation by
the runner. RGB, depth, confidence, intrinsics, and camera-frame roll are
transformed together, preserving the same metric world rays.

## GPU-host preparation

Copy these items to the GPU host:

```text
gluemap/
dataset_audit/manifest_corrected.json
polycam_raw/keyframes/{images,depth,confidence}/
```

Initialize the pinned MapAnything submodule and install GLUEMAP according to
`INSTALL.md`:

```bash
cd /workspace/telluride/gluemap
git submodule update --init thirdparty/mapanything
```

The runner deliberately uses MapAnything directly for this gate. It does not
run SALAD, Doppelgangers++, SIFT, tracking, or global mapping.

## Runs

Pose-conditioned:

```bash
python scripts/run_polycam_mapanything_group.py \
  /workspace/telluride/dataset_audit/manifest_corrected.json \
  configs/telluride_mapanything_groups.json \
  basement_stair_bridge \
  /workspace/telluride/results/basement_stair_bridge/with_pose \
  --source-root /workspace/telluride/polycam_raw \
  --pose-mode corrected \
  --min-confidence 255 \
  --minibatch-size 1 \
  --orientation upright_cw
```

Pose omitted:

```bash
python scripts/run_polycam_mapanything_group.py \
  /workspace/telluride/dataset_audit/manifest_corrected.json \
  configs/telluride_mapanything_groups.json \
  basement_stair_bridge \
  /workspace/telluride/results/basement_stair_bridge/without_pose \
  --source-root /workspace/telluride/polycam_raw \
  --pose-mode none \
  --min-confidence 255 \
  --minibatch-size 1 \
  --orientation upright_cw
```

Both runs retain metric depth and intrinsics. The only intended difference is
whether corrected ARKit camera poses and pose scale are supplied.

The runner enables memory-efficient inference. Minibatch size 1 matches the
lowest-memory configuration in MapAnything's published profiling.

## First bridge result

The first A40 run used MapAnything v1.1.1, BF16, and dense-head minibatch size
1. The initial standalone process took about 71 seconds including setup; the
subsequent single-model-load batch took about 16-19 seconds per 64-view star.
Both outputs retained metric
depth and intrinsics; only pose conditioning differed.

| Measurement | Corrected poses | No poses |
| --- | ---: | ---: |
| Camera distance, frames 2379-3034 | 1.542 m | 0.403 m |
| Camera distance, frames 2380-3034 | 1.563 m | 0.422 m |
| Mean LiDAR depth error | 0.0136 m | 0.0368 m |
| Mean median relative LiDAR error | 0.0079 | 0.0247 |
| Mean metric scaling factor | 1.248 | 1.441 |

The corrected-pose run preserves almost all of the known 1.6 m staircase
trajectory split. Omitting poses removes roughly 1.15 m of that discrepancy,
while retaining mean agreement with confidence-255 LiDAR samples within 3.7
cm. CloudCompare inspection found that the pose-free result is one of the best
basement-landing reconstructions produced during the project.

The resulting project-wide policy is to run MapAnything with RGB, metric depth,
and intrinsics but without pose conditioning. Corrected ARKit poses remain in
the frontend for graph construction and global initialization, and become soft
priors during final refinement.

## Acceptance criteria

Compare the two outputs using:

- predicted versus measured depth at confidence-255 LiDAR pixels;
- predicted camera changes relative to corrected ARKit;
- overlap of the two staircase visits in one coordinate frame;
- metric scaling factor;
- confidence and non-ambiguous masks;
- agreement with the adjacent lower-landing validation group.

Pass if one configuration reconstructs one coherent staircase at metric scale
without damaging locally correct depth surfaces. The pose-free configuration
passed this gate and is now the default for all local groups.
