#!/usr/bin/env python3

import os
import glob
import json
import argparse
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

DEFAULT_EMBEDDING_DIR = '/workspace/data/tcga_embeddings'
DEFAULT_OUTPUT_DIR = '/workspace/data/tcga_spatial_graphs'


def inspect_h5_file(path):
    print(f'\nInspecting: {path}')

    with h5py.File(path, 'r') as f:


        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'{name:30s} shape={obj.shape} dtype={obj.dtype}')
        f.visititems(visitor)


def attr_to_int(attrs, key):
    if key not in attrs:
        return None

    try:
        return int(attrs[key])
    except Exception:
        return None


def get_patient_id(h5_path):
    filename = os.path.basename(h5_path)

    lower = filename.lower()

    if lower.endswith('.hdf5'):
        return filename[:-5]

    if lower.endswith('.h5'):
        return filename[:-3]

    return os.path.splitext(filename)[0]


def load_wsi_metadata_and_coords(path):
    with h5py.File(path, 'r') as f:
        if 'features' not in f:
            raise RuntimeError(f"'features' dataset not found in {path}. Available keys: {list(f.keys())}")

        if 'coords_patching' in f:
            coordinate_key = 'coords_patching'
        elif 'coords' in f:
            coordinate_key = 'coords'
        else:
            raise RuntimeError(f"No coordinate dataset found in {path}. Expected 'coords_patching' or 'coords'. Available keys: {list(f.keys())}")
        feature_shape = tuple(f['features'].shape)

        coords = np.asarray(f[coordinate_key], dtype=np.int64)

        coord_attrs = {key: value for key, value in f[coordinate_key].attrs.items()}

    if len(feature_shape) == 3:
        if feature_shape[0] != 1:
            raise ValueError(f'Unexpected features shape in {path}: {feature_shape}. Expected leading dimension 1.')
        original_num_nodes = int(feature_shape[1])

        embedding_dim = int(feature_shape[2])

        feature_has_leading_dim = True
    elif len(feature_shape) == 2:
        original_num_nodes = int(feature_shape[0])

        embedding_dim = int(feature_shape[1])

        feature_has_leading_dim = False
    else:
        raise ValueError(f'Unexpected features shape in {path}: {feature_shape}')

    if coords.ndim == 3:
        if coords.shape[0] != 1:
            raise ValueError(f'Unexpected coordinate shape in {path}: {coords.shape}')
        coords = coords[0]

    if coords.ndim != 2:
        raise ValueError(f'Coordinates must have shape [N, 2], got {coords.shape} in {path}')

    if coords.shape[1] != 2:
        raise ValueError(f'Coordinates must have exactly 2 columns, got {coords.shape} in {path}')

    if original_num_nodes != coords.shape[0]:
        raise ValueError(f'Embedding count ({original_num_nodes}) does not match coordinate count ({coords.shape[0]}) in {path}')

    if embedding_dim != 1536:
        print(f'WARNING: {path} has embedding dimension {embedding_dim}, expected 1536 for UNI2-h.')

    return (coords, coordinate_key, original_num_nodes, embedding_dim, feature_has_leading_dim, coord_attrs)


def read_feature_rows(h5_path, row_indices, feature_has_leading_dim):
    row_indices = np.asarray(row_indices, dtype=np.int64)

    row_indices = np.sort(row_indices)

    with h5py.File(h5_path, 'r') as f:
        dataset = f['features']

        if feature_has_leading_dim:
            rows = np.asarray(dataset[0, row_indices, :], dtype=np.float32)
        else:
            rows = np.asarray(dataset[row_indices, :], dtype=np.float32)

    return rows


def remove_verified_duplicate_coordinates(h5_path, coords, feature_has_leading_dim, atol=1e-06):
    num_rows = coords.shape[0]

    node_indices = np.arange(num_rows, dtype=np.int64)

    unique_coords, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

    duplicate_groups = np.where(counts > 1)[0]

    if len(duplicate_groups) == 0:
        return (coords, node_indices, 0, 0, 1)
    keep_mask = np.ones(num_rows, dtype=bool)

    duplicate_locations = 0
    duplicate_rows_removed = 0
    maximum_multiplicity = 1

    for group_id in duplicate_groups:
        group_indices = np.where(inverse == group_id)[0]

        multiplicity = len(group_indices)

        maximum_multiplicity = max(maximum_multiplicity, multiplicity)

        duplicate_locations += 1
        features = read_feature_rows(h5_path=h5_path, row_indices=group_indices, feature_has_leading_dim=feature_has_leading_dim)

        reference = features[0]

        if not np.allclose(features, reference[None, :], atol=atol, rtol=0.0):
            coordinate = unique_coords[group_id].tolist()

            raise RuntimeError(f'Duplicate coordinate {coordinate} has {multiplicity} rows with non-identical embeddings. Refusing to deduplicate automatically.')
        keep_mask[group_indices[1:]] = False
        duplicate_rows_removed += multiplicity - 1
    filtered_coords = coords[keep_mask]

    filtered_node_indices = node_indices[keep_mask]

    return (filtered_coords, filtered_node_indices, duplicate_locations, duplicate_rows_removed, maximum_multiplicity)


def estimate_tile_stride(coords):
    if len(coords) < 2:
        return 1.0
    tree = cKDTree(coords)

    distances, _ = tree.query(coords, k=2)

    nearest = distances[:, 1]

    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if len(nearest) == 0:
        raise RuntimeError('Unable to estimate tile spacing because no valid nearest-neighbor distances were found.')
    stride = float(np.median(nearest))

    if stride <= 0:
        raise RuntimeError(f'Estimated invalid tile stride: {stride}')

    return stride


def determine_tile_stride(coords, coord_attrs):
    metadata_stride = attr_to_int(coord_attrs, 'patch_size_level0')

    if metadata_stride is not None and metadata_stride > 0:
        return (float(metadata_stride), 'patch_size_level0')
    stride = estimate_tile_stride(coords)

    return (stride, 'estimated_nearest_neighbor')


def build_spatial_edges(coords, stride, radius_multiplier=1.5):
    num_nodes = coords.shape[0]

    if num_nodes <= 1:
        return (torch.empty((2, 0), dtype=torch.int32), torch.empty((0, 3), dtype=torch.float32))
    radius = stride * radius_multiplier
    coords_float = coords.astype(np.float64)

    tree = cKDTree(coords_float)

    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    if pairs.size == 0:
        return (torch.empty((2, 0), dtype=torch.int32), torch.empty((0, 3), dtype=torch.float32))
    src = pairs[:, 0]

    dst = pairs[:, 1]

    src_bidir = np.concatenate([src, dst])

    dst_bidir = np.concatenate([dst, src])

    edge_index_np = np.stack([src_bidir, dst_bidir], axis=0).astype(np.int32)

    delta = coords_float[dst_bidir] - coords_float[src_bidir]

    distances = np.linalg.norm(delta, axis=1)

    edge_attr_np = np.stack([distances / stride, delta[:, 0] / stride, delta[:, 1] / stride], axis=1).astype(np.float32)

    return (torch.from_numpy(edge_index_np), torch.from_numpy(edge_attr_np))


def create_lightweight_graph(h5_path, radius_multiplier):
    coords, coordinate_key, original_num_nodes, embedding_dim, feature_has_leading_dim, coord_attrs = load_wsi_metadata_and_coords(h5_path)

    patient_id = get_patient_id(h5_path)

    coords, node_indices, duplicate_coordinate_locations, duplicate_rows_removed, maximum_duplicate_multiplicity = remove_verified_duplicate_coordinates(h5_path=h5_path, coords=coords, feature_has_leading_dim=feature_has_leading_dim)

    stride, stride_source = determine_tile_stride(coords=coords, coord_attrs=coord_attrs)

    edge_index, edge_attr = build_spatial_edges(coords=coords, stride=stride, radius_multiplier=radius_multiplier)

    pos = torch.from_numpy(coords.astype(np.int32))

    node_indices_tensor = torch.from_numpy(node_indices.astype(np.int32))

    graph = {'patient_id': patient_id, 'slide_id': patient_id, 'source_h5': h5_path, 'feature_key': 'features', 'coordinate_source': coordinate_key, 'original_num_nodes': int(original_num_nodes), 'num_nodes': int(len(node_indices)), 'embedding_dim': int(embedding_dim), 'node_indices': node_indices_tensor, 'pos': pos, 'edge_index': edge_index, 'edge_attr': edge_attr, 'tile_stride': float(stride), 'stride_source': stride_source, 'radius_multiplier': float(radius_multiplier), 'duplicate_coordinate_locations': int(duplicate_coordinate_locations), 'duplicate_rows_removed': int(duplicate_rows_removed), 'maximum_duplicate_multiplicity': int(maximum_duplicate_multiplicity), 'level0_magnification': attr_to_int(coord_attrs, 'level0_magnification'), 'target_magnification': attr_to_int(coord_attrs, 'target_magnification'), 'patch_size': attr_to_int(coord_attrs, 'patch_size'), 'patch_size_level0': attr_to_int(coord_attrs, 'patch_size_level0')}

    return graph


def validate_graph(graph):
    errors = []

    pos = graph['pos']

    node_indices = graph['node_indices']

    edge_index = graph['edge_index']

    edge_attr = graph['edge_attr']

    num_nodes = graph['num_nodes']

    original_num_nodes = graph['original_num_nodes']

    if pos.ndim != 2:
        errors.append(f'pos must be 2D, got {tuple(pos.shape)}')
    elif pos.shape[1] != 2:
        errors.append(f'pos must have shape [N, 2], got {tuple(pos.shape)}')

    if pos.shape[0] != num_nodes:
        errors.append(f'pos has {pos.shape[0]} rows but num_nodes={num_nodes}')

    if node_indices.ndim != 1:
        errors.append(f'node_indices must be 1D, got {tuple(node_indices.shape)}')

    if len(node_indices) != num_nodes:
        errors.append(f'node_indices has {len(node_indices)} entries but num_nodes={num_nodes}')

    if node_indices.numel() > 0:
        min_source_index = int(node_indices.min().item())

        max_source_index = int(node_indices.max().item())

        if min_source_index < 0:
            errors.append(f'node_indices contains negative index {min_source_index}')

        if max_source_index >= original_num_nodes:
            errors.append(f'node_indices references original H5 row {max_source_index}, but original_num_nodes={original_num_nodes}')
        unique_source_indices = torch.unique(node_indices)

        if len(unique_source_indices) != len(node_indices):
            errors.append('node_indices contains duplicate source rows')

    if edge_index.ndim != 2:
        errors.append('edge_index must be 2D')
    elif edge_index.shape[0] != 2:
        errors.append(f'edge_index must have shape [2, E], got {tuple(edge_index.shape)}')

    if edge_index.numel() > 0:
        min_node = int(edge_index.min().item())

        max_node = int(edge_index.max().item())

        if min_node < 0:
            errors.append(f'edge_index contains negative node index {min_node}')

        if max_node >= num_nodes:
            errors.append(f'edge_index references node {max_node}, but graph contains only {num_nodes} nodes')

    if edge_attr.ndim != 2:
        errors.append('edge_attr must be 2D')
    elif edge_attr.shape[1] != 3:
        errors.append(f'edge_attr must have 3 columns, got {tuple(edge_attr.shape)}')

    if edge_index.ndim == 2 and edge_attr.ndim == 2 and (edge_index.shape[1] != edge_attr.shape[0]):
        errors.append('Number of edges in edge_index does not match number of rows in edge_attr')

    if not os.path.exists(graph['source_h5']):
        errors.append(f"source_h5 does not exist: {graph['source_h5']}")

    return errors


def graph_statistics(graph):
    num_nodes = int(graph['num_nodes'])

    num_edges = int(graph['edge_index'].shape[1])

    average_degree = num_edges / num_nodes if num_nodes > 0 else 0.0

    if num_nodes == 0:
        isolated_nodes = 0
    elif num_edges == 0:
        isolated_nodes = num_nodes
    else:
        degree = torch.bincount(graph['edge_index'][0].long(), minlength=num_nodes)

        isolated_nodes = int((degree == 0).sum().item())

    return {'original_num_nodes': int(graph['original_num_nodes']), 'num_nodes': num_nodes, 'num_edges': num_edges, 'average_degree': average_degree, 'isolated_nodes': isolated_nodes, 'embedding_dim': int(graph['embedding_dim']), 'duplicate_coordinate_locations': int(graph['duplicate_coordinate_locations']), 'duplicate_rows_removed': int(graph['duplicate_rows_removed']), 'maximum_duplicate_multiplicity': int(graph['maximum_duplicate_multiplicity'])}


def atomic_torch_save(obj, output_path):
    temp_path = output_path + '.tmp'

    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        torch.save(obj, temp_path)

        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def existing_graph_is_valid(path):
    if not os.path.exists(path):
        return False

    if os.path.getsize(path) == 0:
        return False

    try:
        graph = torch.load(path, weights_only=False, map_location='cpu')

        if not isinstance(graph, dict):
            return False
        required_keys = {'patient_id', 'source_h5', 'original_num_nodes', 'num_nodes', 'embedding_dim', 'node_indices', 'pos', 'edge_index', 'edge_attr', 'tile_stride', 'stride_source', 'radius_multiplier', 'duplicate_coordinate_locations', 'duplicate_rows_removed'}

        return required_keys.issubset(graph.keys())
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='Create lightweight pathology spatial graphs with verified duplicate-coordinate removal and source-row mapping for UNI2-h embeddings.')

    parser.add_argument('--embedding_dir', type=str, default=DEFAULT_EMBEDDING_DIR)

    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--radius_multiplier', type=float, default=1.5)

    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--inspect_first', action='store_true')

    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.embedding_dir, '*.h5')))

    h5_files += sorted(glob.glob(os.path.join(args.embedding_dir, '*.hdf5')))

    if args.limit is not None:
        h5_files = h5_files[:args.limit]
    print(f'Embedding directory: {args.embedding_dir}')

    print(f'Output directory:    {args.output_dir}')

    print(f'Radius multiplier:   {args.radius_multiplier}')

    print(f'H5 files selected:   {len(h5_files)}')

    if len(h5_files) == 0:
        raise RuntimeError(f'No H5 files found in {args.embedding_dir}')

    if args.inspect_first:
        inspect_h5_file(h5_files[0])
    metadata = []

    successful = 0
    skipped = 0
    failed = 0
    total_duplicate_rows_removed = 0
    slides_with_duplicates = 0

    for h5_path in tqdm(h5_files, desc='Creating lightweight spatial graphs V3'):
        patient_id = get_patient_id(h5_path)

        output_path = os.path.join(args.output_dir, f'{patient_id}.pt')

        if not args.overwrite and existing_graph_is_valid(output_path):
            skipped += 1
            continue

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        try:
            graph = create_lightweight_graph(h5_path=h5_path, radius_multiplier=args.radius_multiplier)

            errors = validate_graph(graph)

            if errors:
                raise RuntimeError('; '.join(errors))
            stats = graph_statistics(graph)

            atomic_torch_save(graph, output_path)

            removed = int(graph['duplicate_rows_removed'])

            total_duplicate_rows_removed += removed

            if removed > 0:
                slides_with_duplicates += 1
                print(f"DEDUPLICATED {patient_id}: removed {removed} duplicate coordinate rows; kept {graph['num_nodes']} spatial nodes from {graph['original_num_nodes']} original H5 rows.")
            metadata.append({'patient_id': patient_id, 'source_h5': h5_path, 'graph_file': output_path, 'coordinate_source': graph['coordinate_source'], 'tile_stride': graph['tile_stride'], 'stride_source': graph['stride_source'], 'radius_multiplier': graph['radius_multiplier'], 'level0_magnification': graph['level0_magnification'], 'target_magnification': graph['target_magnification'], 'patch_size': graph['patch_size'], 'patch_size_level0': graph['patch_size_level0'], **stats})

            successful += 1
        except Exception as exc:
            print(f'FAILED: {h5_path}')

            print(f'Reason: {exc}')

            metadata.append({'patient_id': patient_id, 'source_h5': h5_path, 'status': 'failed', 'error': str(exc)})

            failed += 1
    metadata_path = os.path.join(args.output_dir, 'graph_metadata.json')

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Successful:                  {successful}')

    print(f'Skipped:                     {skipped}')

    print(f'Failed:                      {failed}')

    print(f'Slides with duplicates:      {slides_with_duplicates}')

    print(f'Duplicate rows removed:      {total_duplicate_rows_removed}')

    print(f'Metadata:                    {metadata_path}')


if __name__ == '__main__':
    main()
