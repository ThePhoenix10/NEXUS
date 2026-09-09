#!/usr/bin/env python3

import argparse
import json
import os
import traceback
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

DEFAULT_INPUT_DIR = '/workspace/data/cptac_embeddings'
DEFAULT_OUTPUT_DIR = '/workspace/data/cptac_spatial_graphs'
VALID_STRIDES = (256.0, 512.0, 768.0, 1024.0)


def load_h5_metadata(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise RuntimeError("Missing 'features' dataset")
        features = f['features']

        if features.ndim == 3:
            if features.shape[0] != 1:
                raise RuntimeError(f'Unexpected features shape: {features.shape}')
            n_features = int(features.shape[1])

            embedding_dim = int(features.shape[2])
        elif features.ndim == 2:
            n_features = int(features.shape[0])

            embedding_dim = int(features.shape[1])
        else:
            raise RuntimeError(f'Unexpected features shape: {features.shape}')

        if embedding_dim != 1536:
            raise RuntimeError(f'Unexpected UNI2-h dimension {embedding_dim}; expected 1536')

        if 'coords_patching' in f:
            coords = np.asarray(f['coords_patching'])

            coord_key = 'coords_patching'
        elif 'coords' in f:
            coords = np.asarray(f['coords'])

            coord_key = 'coords'

            if coords.ndim == 3 and coords.shape[0] == 1:
                coords = coords[0]
        else:
            raise RuntimeError('Missing coords_patching/coords')

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise RuntimeError(f'Unexpected coords shape: {coords.shape}')

        if len(coords) != n_features:
            raise RuntimeError(f'Coordinate/feature mismatch: {len(coords)} vs {n_features}')
        preferred_stride = None

        for key in ('patch_size_level0', 'patch_size', 'stride', 'tile_size'):
            if key in f.attrs:
                try:
                    value = float(f.attrs[key])

                    if np.isfinite(value) and value > 0:
                        preferred_stride = value
                        break
                except Exception:
                    pass

    return (coords.astype(np.float32, copy=False), embedding_dim, coord_key, preferred_stride)


def verify_duplicate_embeddings_identical(h5_path, duplicate_groups):
    if not duplicate_groups:
        return

    with h5py.File(h5_path, 'r') as f:
        features = f['features']

        for group in duplicate_groups:
            first = group[0]

            if features.ndim == 3:
                reference = np.asarray(features[0, first, :])

                for idx in group[1:]:
                    candidate = np.asarray(features[0, idx, :])

                    if not np.array_equal(reference, candidate):
                        raise RuntimeError(f'Duplicate coordinate rows have different embeddings: {group[:10]}')
            else:
                reference = np.asarray(features[first, :])

                for idx in group[1:]:
                    candidate = np.asarray(features[idx, :])

                    if not np.array_equal(reference, candidate):
                        raise RuntimeError(f'Duplicate coordinate rows have different embeddings: {group[:10]}')


def deduplicate_coordinates(coords, h5_path):
    lookup = {}

    for idx, xy in enumerate(coords):
        key = (float(xy[0]), float(xy[1]))

        lookup.setdefault(key, []).append(idx)
    duplicate_groups = [v for v in lookup.values() if len(v) > 1]

    verify_duplicate_embeddings_identical(h5_path, duplicate_groups)

    keep_indices = np.array(sorted((v[0] for v in lookup.values())), dtype=np.int64)

    dedup_coords = coords[keep_indices]

    removed = int(len(coords) - len(dedup_coords))

    return (dedup_coords, keep_indices, duplicate_groups, removed)


def estimate_stride(coords, preferred_stride=None):
    if preferred_stride is not None:
        closest = min(VALID_STRIDES, key=lambda x: abs(x - float(preferred_stride)))

        if abs(float(preferred_stride) - closest) <= max(1.0, 0.05 * closest):
            return (float(closest), 'h5_metadata')

    if len(coords) < 2:
        return (256.0, 'fallback_single_node')
    tree = cKDTree(coords)

    distances, _ = tree.query(coords, k=2, workers=-1)

    nearest = distances[:, 1]

    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if len(nearest) == 0:
        return (256.0, 'fallback_no_positive_nn')
    median_nn = float(np.median(nearest))

    closest = min(VALID_STRIDES, key=lambda x: abs(x - median_nn))

    if abs(median_nn - closest) <= 0.3 * closest:
        return (float(closest), 'nearest_neighbor')

    return (median_nn, 'nearest_neighbor_unrounded')


def build_radius_graph(coords, stride, radius_multiplier):
    if len(coords) == 0:
        raise RuntimeError('Zero-node graph')
    radius = float(stride) * float(radius_multiplier)

    tree = cKDTree(coords)

    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    if pairs.size == 0:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32), radius)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    src = pairs[:, 0]

    dst = pairs[:, 1]

    delta = coords[dst] - coords[src]

    distance = np.sqrt(np.sum(delta * delta, axis=1))

    forward_attr = np.column_stack((distance / stride, delta[:, 0] / stride, delta[:, 1] / stride)).astype(np.float32)

    reverse_attr = np.column_stack((distance / stride, -delta[:, 0] / stride, -delta[:, 1] / stride)).astype(np.float32)

    edge_index = np.vstack((np.concatenate((src, dst)), np.concatenate((dst, src)))).astype(np.int64)

    edge_attr = np.concatenate((forward_attr, reverse_attr), axis=0)

    return (torch.from_numpy(edge_index).long(), torch.from_numpy(edge_attr).float(), radius)


def output_name(h5_path, input_root):
    relative = h5_path.relative_to(input_root)

    if len(relative.parts) == 1:
        return h5_path.stem + '.pt'
    prefix = '__'.join(relative.parts[:-1])

    return f'{prefix}__{h5_path.stem}.pt'


def process_one(h5_path, input_root, output_root, radius_multiplier, overwrite):
    output_path = output_root / output_name(h5_path, input_root)

    if output_path.exists() and (not overwrite):
        return {'status': 'skipped_existing', 'h5': str(h5_path), 'graph': str(output_path)}
    coords, embedding_dim, coord_key, preferred_stride = load_h5_metadata(h5_path)

    original_n = int(len(coords))

    coords, node_indices, duplicate_groups, removed = deduplicate_coordinates(coords, h5_path)

    stride, stride_source = estimate_stride(coords, preferred_stride)

    edge_index, edge_attr, radius = build_radius_graph(coords, stride, radius_multiplier)

    n_nodes = int(len(coords))

    n_edges = int(edge_index.shape[1])

    graph = {'source_h5': str(h5_path.resolve()), 'feature_key': 'features', 'coord_key': coord_key, 'pos': torch.from_numpy(coords).float(), 'edge_index': edge_index, 'edge_attr': edge_attr, 'node_indices': torch.from_numpy(node_indices).long(), 'embedding_dim': int(embedding_dim), 'original_num_nodes': original_n, 'num_nodes': n_nodes, 'num_directed_edges': n_edges, 'stride': float(stride), 'stride_source': stride_source, 'radius_multiplier': float(radius_multiplier), 'radius': float(radius), 'duplicate_rows_removed': int(removed), 'lightweight': True, 'dataset': 'CPTAC', 'embedding_model': 'UNI2-h'}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(graph, output_path)

    return {'status': 'created', 'h5': str(h5_path), 'graph': str(output_path), 'num_nodes': n_nodes, 'original_num_nodes': original_n, 'num_directed_edges': n_edges, 'avg_degree': float(n_edges / n_nodes) if n_nodes else 0.0, 'stride': float(stride), 'stride_source': stride_source, 'radius': float(radius), 'duplicate_rows_removed': int(removed), 'duplicate_groups': int(len(duplicate_groups))}


def make_summary(results):
    created = [r for r in results if r.get('status') == 'created']

    skipped = [r for r in results if r.get('status') == 'skipped_existing']

    failed = [r for r in results if r.get('status') == 'failed']

    summary = {'total_requested': len(results), 'created': len(created), 'skipped_existing': len(skipped), 'failed': len(failed)}

    if created:
        nodes = np.array([r['num_nodes'] for r in created], dtype=float)

        degrees = np.array([r['avg_degree'] for r in created], dtype=float)

        removed = np.array([r['duplicate_rows_removed'] for r in created], dtype=int)

        summary.update({'embedding_dim': 1536, 'node_count_min': int(nodes.min()), 'node_count_median': float(np.median(nodes)), 'node_count_max': int(nodes.max()), 'avg_degree_min': float(degrees.min()), 'avg_degree_median': float(np.median(degrees)), 'avg_degree_max': float(degrees.max()), 'graphs_with_duplicates_removed': int(np.sum(removed > 0)), 'duplicate_rows_removed_total': int(removed.sum()), 'stride_counts': dict(Counter((str(r['stride']) for r in created)))})

    return summary


def main():
    parser = argparse.ArgumentParser(description='Create lightweight spatial graphs for CPTAC UNI2-h embeddings.')

    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)

    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--radius-multiplier', type=float, default=1.5)

    parser.add_argument('--limit', type=int, default=None)

    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    input_root = Path(args.input_dir)

    output_root = Path(args.output_dir)

    if not input_root.exists():
        raise FileNotFoundError(f'Input embedding directory does not exist: {input_root}')
    output_root.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_root.rglob('*.h5'))

    if args.limit is not None:
        h5_files = h5_files[:args.limit]

    if not h5_files:
        raise RuntimeError(f'No H5 files found under {input_root}')
    print(f'Input:              {input_root}')

    print(f'Output:             {output_root}')

    print(f'H5 files:           {len(h5_files)}')

    print(f'Radius multiplier:  {args.radius_multiplier}')

    results = []

    for h5_path in tqdm(h5_files, desc='Building CPTAC graphs', unit='slide'):
        try:
            result = process_one(h5_path, input_root, output_root, args.radius_multiplier, args.overwrite)
        except Exception as exc:
            result = {'status': 'failed', 'h5': str(h5_path), 'error': str(exc), 'traceback': traceback.format_exc()}
        results.append(result)
    summary = make_summary(results)

    manifest_path = output_root / 'cptac_spatial_graph_manifest.json'
    qc_path = output_root / 'cptac_spatial_graph_qc.json'
    failure_path = output_root / 'cptac_spatial_graph_failures.json'

    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open(qc_path, 'w') as f:
        json.dump(summary, f, indent=2)
    failures = [r for r in results if r.get('status') == 'failed']

    with open(failure_path, 'w') as f:
        json.dump(failures, f, indent=2)

    for key, value in summary.items():
        print(f'{key}: {value}')
    print(f'QC:       {qc_path}')

    print(f'Manifest: {manifest_path}')

    print(f'Failures: {failure_path}')

    print(f'Graphs:   {output_root}')


if __name__ == '__main__':
    main()
