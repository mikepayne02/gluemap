# Telluride pose/depth graph

The post-reset frontend follows phases 4 and 7 of the reconstruction handoff,
with one evidence-driven change: corrected ARKit poses are not supplied to
MapAnything.

Build the frontend graph with:

```bash
PYTHONPATH=. python scripts/build_polycam_pose_depth_groups.py \
  /path/to/manifest_corrected.json \
  /path/to/polycam_raw \
  configs/telluride_postreset_pose_depth_groups.json \
  /path/to/postreset_pose_depth_edges.csv \
  --start 2198 \
  --end 3119 \
  --verified-config configs/telluride_mapanything_groups.json \
  --group-size 64 \
  --minimum-memberships 2 \
  --temporal-neighbors 8 \
  --spatial-radius-m 3.5 \
  --depth-pixel-step 8
```

The frontend always adds local temporal edges. Non-temporal candidates come
from corrected-ARKit camera proximity and must pass bidirectional projection of
confidence-255 LiDAR samples into both measured depth maps. Manually verified
staircase and landing correspondences are forced edges. Each generated star
contains direct neighbors of its anchor; transitive graph walks cannot pull
several indirectly connected rooms into one MapAnything group.

The post-reset graph contains 7,340 temporal edges, 16,691
depth-supported spatial edges, three verified edges, and 44 groups. Every frame
appears in at least two groups.

The stricter full-house graph uses 153 groups and 86,459 frontend edges (24,924
temporal, 61,532 bidirectional depth-overlap, and three verified). Its
927-edge shared-star graph is connected and every one of the 3,120 frames is a
member of at least two stars.

Run all groups with one model load:

```bash
PYTHONPATH=. python scripts/run_polycam_mapanything_groups.py \
  /path/to/manifest_corrected.json \
  configs/telluride_postreset_pose_depth_groups.json \
  /path/to/results/postreset_posefree_graph \
  --source-root /path/to/polycam_raw \
  --minibatch-size 1
```

The batch is resumable and retains corrected ARKit cameras as metadata, but
always omits them from MapAnything inference.

After all groups complete, solve group transforms and export one frame per
camera into a fused cloud:

```bash
PYTHONPATH=.:scripts python scripts/align_polycam_mapanything_groups.py \
  /path/to/manifest_corrected.json \
  configs/telluride_postreset_pose_depth_groups.json \
  /path/to/results/postreset_posefree_graph \
  /path/to/results/postreset_aligned \
  --source-root /path/to/polycam_raw \
  --pixel-step 4
```

For metric, orientation-preserving fusion, reject group-pair measurements whose
initial shared-camera medians exceed 0.5 m or 15 degrees, retain the minimum
ranked rejected edges needed for graph connectivity, and keep scale and ARKit
orientation strongly regularized:

```bash
  --shared-position-sigma-m 0.05 \
  --shared-rotation-sigma-deg 10 \
  --arkit-translation-sigma-m 5 \
  --arkit-rotation-sigma-deg 10 \
  --scale-prior-sigma 0.001 \
  --max-edge-position-median-m 0.5 \
  --max-edge-rotation-median-deg 15
```

`--skip-cloud` runs alignment diagnostics without spending time decoding every
image and writing a dense PLY. Use it for weight selection, then export only the
selected solve.

This is the first, camera/group-transform-only stage of the handoff's global
solve. Shared predicted cameras align overlapping stars. Corrected ARKit poses
initialize the global gauge and weakly regularize group transforms. Neural
points, measured-depth residuals, gravity, temporal smoothness, and small
per-camera corrections remain subsequent optimization stages.

On the 922-frame post-reset segment, the first metric filtered solve retained
128 clean edges and added five rejected edges as a minimal connectivity
backbone. Its retained shared-camera median improved from 13.5 cm to 6.4 cm,
scale stayed within 0.987-1.060, and the median optimized-camera displacement
from corrected ARKit was 14.1 cm. An unfiltered bridge-favoring candidate is
also retained because it closes the manually verified staircase more strongly;
the two are evaluated separately rather than hiding that tradeoff in one score.

## First full-house result

All 153 pose-free stars completed on an A40 with no failed groups. After the
one-time model load, a 64-view star took about 16-19 seconds and peaked at 12.5
GB allocated VRAM. The saved predictions cover all 3,120 frames.

The primary full-house solve retained 541 initially consistent shared-star
edges and added 17 least-bad rejected edges as a minimum connectivity backbone.
Across those 558 edges, optimization changed the shared-camera median from
10.1 cm to 5.3 cm, position p95 from 49.2 cm to 40.8 cm, and rotation p95 from
11.4 to 9.5 degrees. Median camera displacement from corrected ARKit was 11.5
cm. Group scale ranged from 0.935 to 1.077.

An all-edge alternative is retained for comparison. It closes the manually
verified staircase more strongly (about 0.86 m versus 1.24 m in the filtered
solution), but its house-wide position p95 remains 2.29 m, rotation p95 remains
89 degrees, and one group reaches 1.316 scale. It is therefore not the primary
Nerfstudio export.

The selected dense preview has 37,929,490 colored points. The accompanying
Nerfstudio export contains all 3,120 optimized cameras and references that PLY.
COLMAP text cameras and images are also emitted; `points3D.txt` is deliberately
empty because dense neural samples are not feature-track observations.
