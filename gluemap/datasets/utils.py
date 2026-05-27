import argparse
import csv
import glob
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_image_list(images_path: str) -> list[str]:
    """List image files under ``images_path`` recursively, sorted by path.

    Filters by extension (png, jpg, jpeg, bmp, tiff) — case-insensitive.

    Args:
        images_path: Root directory to search.

    Returns:
        Absolute paths of every image file under ``images_path``, in
        lexicographic order.
    """
    img_list = sorted(glob.glob(images_path + "/**", recursive=True))
    # keep only the images
    img_list = [
        x
        for x in img_list
        if os.path.isfile(x)
        and x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    return img_list


def establish_neighbors_sequential(
    image_names: list[str],
    num_neighbors: int = 30,
    return_raw: bool = False,
) -> list[list[int]] | torch.Tensor:
    """Build a sliding-window neighbor list for sequentially ordered images.

    For each image index ``i``, picks ``num_neighbors`` distinct neighbors
    centered on ``i`` (half on each side), padding with extra neighbors
    from the available side when the window hits a sequence boundary.

    Args:
        image_names: Image identifiers in temporal/sequential order; only
            the length is used.
        num_neighbors: Number of neighbors per image. Capped at
            ``len(image_names) - 1``.
        return_raw: If ``True``, return the per-image neighbor lists as
            plain Python lists (length ``num_neighbors``). If ``False``,
            return a single ``torch.Tensor`` whose first column is the
            center index ``i``.

    Returns:
        Either a list of ``num_neighbors``-length neighbor lists per
        image (``return_raw=True``) or a ``(N, num_neighbors + 1)``
        ``torch.Tensor`` of long indices with the center index in
        column 0 (``return_raw=False``).
    """
    N = len(image_names)
    num_neighbors = min(num_neighbors, N - 1)

    neighbors_all = []
    for i in range(N):
        neighbors = []
        left = max(0, i - num_neighbors // 2)
        right = min(N, i + num_neighbors // 2 + 1)

        for j in range(left, right):
            if j != i:
                neighbors.append(j)

        # If not enough neighbors, pad from the other side
        while len(neighbors) < num_neighbors:
            if left > 0:
                left -= 1
                neighbors.insert(0, left)
            elif right < N:
                neighbors.append(right)
                right += 1
            else:
                break

        neighbors_all.append(neighbors)

    if return_raw:
        return neighbors_all

    neighbors_ori = torch.Tensor(neighbors_all).long()
    neighbors = torch.cat(
        [torch.arange(N).unsqueeze(1).to(neighbors_ori.device), neighbors_ori],
        dim=1,
    )

    return neighbors


def retrieve_global_neighbors(
    args: argparse.Namespace,
    neighbors_sequential: list[torch.Tensor],
    descriptors_all: list[torch.Tensor],
) -> np.ndarray:
    """Combine sequential neighbors with FAISS-retrieved global neighbors.

    Each per-sequence ``neighbors_sequential[s]`` tensor is a
    ``(N_s, K + 1)`` matrix with the center index in column 0 (as
    produced by :func:`establish_neighbors_sequential`). Global neighbors
    are obtained by an L2 search over the concatenated descriptor bank
    in ``descriptors_all`` and merged with the local neighbors,
    deduplicated, and returned as undirected unique pairs ``(i, j)``
    with ``i < j``.

    Args:
        args: Parsed CLI/config namespace; reads ``num_neighbors`` and
            ``num_neighbors_sequential``. Setting ``num_neighbors <= 0``
            disables the FAISS retrieval step (sequential neighbors only).
        neighbors_sequential: One tensor per sequence, each of shape
            ``(N_s, K_s + 1)`` with column 0 the center index.
        descriptors_all: One descriptor tensor per sequence, each of
            shape ``(N_s, D)``. Concatenated row-wise to form the FAISS
            index.

    Returns:
        A ``(P, 2)`` array of undirected pair indices ``(i, j)`` with
        ``i < j`` and no duplicates.
    """
    logger.info("Establishing global neighbors...")
    neighbors_local = [
        x[i].cpu().numpy()
        for x in neighbors_sequential
        for i in range(x.shape[0])
    ]
    if args.num_neighbors > 0:
        descriptors_db = torch.cat(descriptors_all, dim=0)
        embed_size = descriptors_db.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)
        faiss_index.add(descriptors_db)  # add vectors to the index

        chosen_indexes = np.arange(descriptors_db.shape[0])
        sampled_neighbors = [
            set(neighbors_local[i].tolist())
            for i in range(len(neighbors_local))
        ]
        distance_extend, predictions_extend = faiss_index.search(
            descriptors_db[chosen_indexes],
            args.num_neighbors + args.num_neighbors_sequential + 1,
        )

        # exclude itself and invalid FAISS results (-1 when k > index size)
        predictions_extend = predictions_extend[:, 1:]

        # exclude invalid FAISS results (-1) and ones already in neighbors_local
        chosen_indexes_item = [
            [
                j
                for j, x in enumerate(predictions_extend[i].tolist())
                if x >= 0 and x not in sampled_neighbors[i]
            ][: args.num_neighbors]
            for i in range(len(predictions_extend))
        ]

        # Build retrieval pairs directly
        # (rows may have different lengths after filtering)
        pairs_retrieval = (
            np.array(
                [
                    (i, predictions_extend[i][j])
                    for i in range(len(predictions_extend))
                    for j in chosen_indexes_item[i]
                ],
                dtype=int,
            ).reshape(-1, 2)
            if any(len(c) > 0 for c in chosen_indexes_item)
            else np.zeros((0, 2), dtype=int)
        )
    else:
        pairs_retrieval = np.zeros((0, 2), dtype=int)

    pairs_local = np.concatenate(
        [
            np.stack(
                [
                    np.ones(neighbors_local[i].shape, dtype=int) * i,
                    neighbors_local[i],
                ],
                axis=-1,
            )
            for i in range(len(neighbors_local))
        ],
        axis=0,
    )

    pairs = np.array(
        sorted(
            list(
                set(tuple(x) for x in pairs_retrieval).union(
                    tuple(x) for x in pairs_local
                )
            )
        )
    )

    # collect back and forth
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    pairs = pairs[
        (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] < pairs[:, 1])
    ]  # only keep valid pairs where the first index is less than the second
    pairs = np.unique(pairs, axis=0)  # remove duplicates
    return pairs


def load_pair_graph(
    pair_graph_path: str,
    image_names: list[str],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Load a deterministic pair graph for a dataset.

    The graph file can be:

    - JSON list of pairs, e.g. ``[[0, 1], ["0000.jpg", "0040.jpg"]]``
    - JSON object with ``pairs`` and optional ``sequential_edges``
    - CSV/TXT lines with two columns separated by comma or whitespace

    Pair endpoints may be local image indices or image basenames/relative
    paths. Returned pairs are undirected, unique, and use local dataset
    indices.
    """
    path = Path(pair_graph_path)
    name_to_idx = {name: i for i, name in enumerate(image_names)}
    base_to_idx = {Path(name).name: i for i, name in enumerate(image_names)}

    def resolve(value) -> int:
        if isinstance(value, int):
            idx = value
        elif isinstance(value, float) and value.is_integer():
            idx = int(value)
        elif isinstance(value, str):
            token = value.strip()
            if token.isdigit():
                idx = int(token)
            elif token in name_to_idx:
                idx = name_to_idx[token]
            elif Path(token).name in base_to_idx:
                idx = base_to_idx[Path(token).name]
            else:
                raise KeyError(f"Unknown image in pair graph: {value!r}")
        else:
            raise TypeError(f"Unsupported pair endpoint: {value!r}")

        if idx < 0 or idx >= len(image_names):
            raise IndexError(f"Pair graph index out of range: {idx}")
        return idx

    def normalize(raw_pairs) -> list[tuple[int, int]]:
        pairs_out = []
        for raw in raw_pairs:
            if isinstance(raw, dict):
                a = raw.get("i", raw.get("image0", raw.get("src")))
                b = raw.get("j", raw.get("image1", raw.get("dst")))
            else:
                a, b = raw[:2]
            i, j = resolve(a), resolve(b)
            if i == j:
                continue
            pairs_out.append((min(i, j), max(i, j)))
        return sorted(set(pairs_out))

    sequential_edges: list[tuple[int, int]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            pairs = normalize(data.get("pairs", []))
            if "sequential_edges" in data:
                sequential_edges = normalize(data["sequential_edges"])
        else:
            pairs = normalize(data)
    else:
        rows = []
        with path.open(newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "," if "," in sample else None
            if delimiter:
                reader = csv.reader(f)
                rows = [row for row in reader if row and not row[0].startswith("#")]
            else:
                rows = [
                    line.split()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        pairs = normalize(rows)

    return np.asarray(pairs, dtype=int).reshape(-1, 2), sequential_edges
