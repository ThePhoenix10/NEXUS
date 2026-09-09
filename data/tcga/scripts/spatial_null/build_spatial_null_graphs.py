#!/usr/bin/env python3

import argparse
import json
import math
import zlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

DEFAULT_INPUT_DIR = '/workspace/data/tcga_spatial_graphs'
DEFAULT_OUTPUT_DIR = '/workspace/data/spatial_null_graphs'
DEFAULT_SEED = 42
DEFAULT_RADIUS_MULTIPLIER = 1.5


def patient_seed(patient_id, base_seed):
    crc = zlib.crc32(str(patient_id).encode('utf-8'))

    return (int(base_seed) + int(crc)) % 2 ** 32


def to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    return np.asarray(x).reshape(-1)


def to_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)

    if arr.ndim != 2:
        raise RuntimeError(f'Expected 2D array, got shape {arr.shape}')

    return arr


def get_scalar(graph, key, default=None):
    if key not in graph:
        return default
    value = graph[key]

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()

        return value

    return value


def build_radius_graph_from_positions(pos, tile_stride, radius_multiplier):
    n = int(pos.shape[0])

    if n == 0:
        raise RuntimeError('Graph has zero nodes.')

    if n == 1:
        edge_index = np.zeros((2, 0), dtype=np.int32)

        edge_attr = np.zeros((0, 3), dtype=np.float32)

        return (edge_index, edge_attr)
    radius = float(tile_stride) * float(radius_multiplier)

    tree = cKDTree(pos.astype(np.float64))

    undirected_pairs = sorted(tree.query_pairs(r=radius, output_type='set'))

    src_list = []

    dst_list = []

    attr_list = []

    for i, j in undirected_pairs:
        dx_ij = float(pos[j, 0] - pos[i, 0])

        dy_ij = float(pos[j, 1] - pos[i, 1])

        dist_ij = math.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)

        norm_dist = dist_ij / float(tile_stride)

        norm_dx = dx_ij / float(tile_stride)

        norm_dy = dy_ij / float(tile_stride)

        src_list.append(i)

        dst_list.append(j)

        attr_list.append([norm_dist, norm_dx, norm_dy])

        src_list.append(j)

        dst_list.append(i)

        attr_list.append([norm_dist, -norm_dx, -norm_dy])
    edge_index = np.asarray([src_list, dst_list], dtype=np.int32)

    edge_attr = np.asarray(attr_list, dtype=np.float32)

    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)

    if edge_attr.size == 0:
        edge_attr = np.zeros((0, 3), dtype=np.float32)

    return (edge_index, edge_attr)


def shuffle_tile_placements(input_path, output_path, base_seed):
    graph = torch.load(input_path, map_location='cpu', weights_only=False)

    required = ['patient_id', 'node_indices', 'pos', 'tile_stride']

    missing = [k for k in required if k not in graph]

    if missing:
        raise RuntimeError(f'{input_path.name} missing required keys: {missing}')
    patient_id = str(graph['patient_id'])

    node_indices = to_numpy_1d(graph['node_indices'])

    pos = to_numpy_2d(graph['pos']).astype(np.int32)

    n = int(len(node_indices))

    if int(pos.shape[0]) != n:
        raise RuntimeError(f'{input_path.name}: node_indices has {n} rows but pos has {pos.shape[0]} rows.')
    tile_stride = float(get_scalar(graph, 'tile_stride'))

    radius_multiplier = float(get_scalar(graph, 'radius_multiplier', DEFAULT_RADIUS_MULTIPLIER))

    if n <= 1:
        shuffled_pos = pos.copy()

        permutation = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(patient_seed(patient_id, base_seed))

        permutation = rng.permutation(n)

        if np.array_equal(permutation, np.arange(n)):
            permutation = np.roll(permutation, 1)
        shuffled_pos = pos[permutation].copy()
    edge_index, edge_attr = build_radius_graph_from_positions(shuffled_pos, tile_stride=tile_stride, radius_multiplier=radius_multiplier)

    out = dict(graph)

    if torch.is_tensor(graph['pos']):
        out['pos'] = torch.as_tensor(shuffled_pos, dtype=graph['pos'].dtype)
    else:
        out['pos'] = shuffled_pos

    if torch.is_tensor(graph['edge_index']):
        out['edge_index'] = torch.as_tensor(edge_index, dtype=graph['edge_index'].dtype)
    else:
        out['edge_index'] = edge_index

    if torch.is_tensor(graph['edge_attr']):
        out['edge_attr'] = torch.as_tensor(edge_attr, dtype=graph['edge_attr'].dtype)
    else:
        out['edge_attr'] = edge_attr
    out['spatial_control'] = 'within_slide_tile_placement_permutation'
    out['spatial_control_seed'] = int(base_seed)

    out['spatial_control_patient_seed'] = int(patient_seed(patient_id, base_seed))

    out['spatial_control_method'] = 'Each tile kept its original UNI2-h embedding identity (node_indices unchanged), but tile coordinates were randomly permuted across nodes within the same patient. A new radius-neighborhood graph was then rebuilt from the permuted coordinates, and edge_attr [distance, dx, dy] was recomputed. This destroys true spatial tile placement while preserving morphology content, node count, patient identity, and the same radius-graph construction rule.'
    out['spatial_control_changed_coordinates'] = True
    out['spatial_control_changed_edges'] = True
    out['spatial_control_changed_edge_attr'] = True
    out['spatial_control_changed_node_indices'] = False
    out['spatial_control_permutation'] = torch.as_tensor(permutation, dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(out, output_path)

    changed_fraction = float(np.mean(np.any(shuffled_pos != pos, axis=1))) if n > 0 else 0.0

    return {'patient_id': patient_id, 'num_nodes': n, 'num_edges': int(edge_index.shape[1]), 'changed_fraction_coordinates': changed_fraction, 'tile_stride': tile_stride, 'radius_multiplier': radius_multiplier}


def main():
    parser = argparse.ArgumentParser(description='Create spatial-destruction control graphs by keeping each tile embedding with its tile while randomly permuting tile placements (coordinates) within each patient and rebuilding the radius graph.')

    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)

    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)

    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir)

    graph_paths = sorted(input_dir.glob('*.pt'))

    if not graph_paths:
        raise RuntimeError(f'No .pt graphs found in {input_dir}')

    if args.limit is not None:
        graph_paths = graph_paths[:args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Input graph directory:  {input_dir}')

    print(f'Output graph directory: {output_dir}')

    print(f'Graphs selected:         {len(graph_paths):,}')

    print(f'Base seed:               {args.seed}')

    successful = 0
    skipped = 0
    failed = []

    total_nodes = 0
    total_edges = 0
    changed_fractions = []

    for input_path in tqdm(graph_paths, desc='Creating shuffled-placement graphs', unit='graph'):
        output_path = output_dir / input_path.name

        if output_path.exists() and (not args.overwrite):
            skipped += 1
            continue

        try:
            info = shuffle_tile_placements(input_path=input_path, output_path=output_path, base_seed=args.seed)

            successful += 1
            total_nodes += info['num_nodes']

            total_edges += info['num_edges']

            changed_fractions.append(info['changed_fraction_coordinates'])
        except Exception as exc:
            failed.append({'file': input_path.name, 'error': str(exc)})
    metadata = {'control': 'within_slide_tile_placement_permutation', 'base_seed': int(args.seed), 'input_dir': str(input_dir), 'output_dir': str(output_dir), 'selected': int(len(graph_paths)), 'successful': int(successful), 'skipped': int(skipped), 'failed': int(len(failed)), 'total_nodes_processed': int(total_nodes), 'total_edges_processed': int(total_edges), 'mean_fraction_coordinates_reassigned': float(np.mean(changed_fractions)) if changed_fractions else None, 'preserved': ['patient identity', 'source H5', 'tile embedding identity', 'node_indices', 'node count', 'radius-graph construction rule'], 'perturbed': ['tile coordinates / placement', 'edge_index', 'edge_attr', 'spatial neighborhood relationships'], 'method': 'Within each patient, tile coordinates were randomly permuted across nodes while node_indices were kept unchanged. A new radius graph was rebuilt from the permuted coordinates, and edge_attr [distance, dx, dy] was recomputed. This keeps each UNI2-h embedding attached to its original tile while destroying the true placement of tiles in tissue space.', 'failures': failed}

    metadata_path = output_dir / 'shuffle_metadata.json'

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Successful: {successful:,}')

    print(f'Skipped:    {skipped:,}')

    print(f'Failed:     {len(failed):,}')

    if changed_fractions:
        print(f'Mean fraction of tiles moved to a new coordinate: {np.mean(changed_fractions):.6f}')
    print(f'Metadata:   {metadata_path}')

    if failed:
        raise RuntimeError(f'{len(failed)} graphs failed. See {metadata_path}')


if __name__ == '__main__':
    main()
