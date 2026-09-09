#!/usr/bin/env python3

import os
import json
import random
import argparse
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix, classification_report

DEFAULT_GRAPH_DIR = '/workspace/data/spatial_null_graphs'
DEFAULT_FUSED_ROWS = '/workspace/data/patient_fused_matrix_rows.txt'
DEFAULT_CASE_MAP = '/workspace/metadata/case_id_barcode_map.csv'
DEFAULT_LABELS_CSV = '/workspace/metadata/tcga_labels.csv'
DEFAULT_OUTPUT = '/workspace/results/spatial_null'
DEFAULT_SPLIT_DIR = '/workspace/splits'


def seed_everything(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def normalize_patient_id(x):
    x = str(x).strip()

    if x.endswith('.pt'):
        x = x[:-3]

    if x.endswith('.h5'):
        x = x[:-3]
    parts = x.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3])

    return x


def extract_tcga_site(patient_id):
    patient_id = normalize_patient_id(patient_id)

    parts = patient_id.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return parts[1]

    return 'UNKNOWN'


def make_site_stratified_folds(df, n_splits, seed):
    work = df.reset_index(drop=True).copy()

    labels = work['label_text'].astype(str).tolist()

    sites = work['site'].astype(str).tolist()

    rng = np.random.default_rng(seed)

    label_values = sorted(set(labels))

    site_values = sorted(set(sites))

    label_to_idx = {value: i for i, value in enumerate(label_values)}

    site_to_idx = {value: i for i, value in enumerate(site_values)}

    label_totals = np.zeros(len(label_values), dtype=np.int64)

    site_totals = np.zeros(len(site_values), dtype=np.int64)

    for label in labels:
        label_totals[label_to_idx[label]] += 1

    for site in sites:
        site_totals[site_to_idx[site]] += 1
    strata = {}

    for row_idx, (label, site) in enumerate(zip(labels, sites)):
        strata.setdefault((label, site), []).append(row_idx)
    stratum_items = list(strata.items())

    rng.shuffle(stratum_items)

    stratum_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_sizes = np.zeros(n_splits, dtype=np.int64)

    fold_label_counts = np.zeros((n_splits, len(label_values)), dtype=np.int64)

    fold_site_counts = np.zeros((n_splits, len(site_values)), dtype=np.int64)

    assignments = np.full(len(work), -1, dtype=np.int64)

    target_size = len(work) / n_splits

    for (label, site), row_indices in stratum_items:
        row_indices = np.asarray(row_indices, dtype=np.int64)

        rng.shuffle(row_indices)

        label_idx = label_to_idx[label]

        site_idx = site_to_idx[site]

        target_label = label_totals[label_idx] / n_splits
        target_site = site_totals[site_idx] / n_splits

        for row_idx in row_indices:
            label_scores = fold_label_counts[:, label_idx] / max(target_label, 1.0)

            site_scores = fold_site_counts[:, site_idx] / max(target_site, 1.0)

            size_scores = fold_sizes / max(target_size, 1.0)

            scores = 2.0 * label_scores + 1.0 * site_scores + 0.25 * size_scores
            minimum = scores.min()

            candidates = np.flatnonzero(np.isclose(scores, minimum, rtol=0.0, atol=1e-12))

            chosen = int(rng.choice(candidates))

            assignments[row_idx] = chosen
            fold_sizes[chosen] += 1
            fold_label_counts[chosen, label_idx] += 1
            fold_site_counts[chosen, site_idx] += 1

    if np.any(assignments < 0):
        raise RuntimeError('Site-stratified fold assignment left unassigned patients.')

    return assignments


def validate_site_stratification(df, fold_column, n_splits):
    all_classes = set(df['label_text'].astype(str))

    all_sites = set(df['site'].astype(str))

    for fold in range(n_splits):
        fold_df = df.loc[df[fold_column] == fold]

        missing_classes = all_classes - set(fold_df['label_text'].astype(str))

        print(f"Fold {fold + 1}: n={len(fold_df):,} | classes={fold_df['label_text'].nunique()}/{len(all_classes)} | sites={fold_df['site'].nunique()}/{len(all_sites)} | missing_classes={len(missing_classes)}")

        if missing_classes:
            print('  Missing classes:', sorted(missing_classes))


def load_predefined_split(split_dir, fold_number):
    split_path = Path(split_dir) / f'fold_{fold_number}' / 'split.csv'

    if not split_path.exists():
        raise FileNotFoundError(f'Missing pre-generated split file: {split_path}')
    split_df = pd.read_csv(split_path, keep_default_na=False)

    required = {'patient_norm', 'label_text', 'site', 'graph_path', 'split'}

    missing = required - set(split_df.columns)

    if missing:
        raise RuntimeError(f'{split_path} is missing required columns: {sorted(missing)}')
    split_df = split_df.copy()

    split_df['patient_norm'] = split_df['patient_norm'].map(normalize_patient_id)

    train_df = split_df.loc[split_df['split'] == 'train'].copy().reset_index(drop=True)

    val_df = split_df.loc[split_df['split'] == 'val'].copy().reset_index(drop=True)

    test_df = split_df.loc[split_df['split'] == 'test'].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f'Invalid pre-generated split {split_path}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
    shuffled_graph_dir = Path(DEFAULT_GRAPH_DIR)

    for frame in [train_df, val_df, test_df, split_df]:
        frame['graph_path'] = frame['patient_norm'].map(lambda patient: str(shuffled_graph_dir / f'{patient}.pt'))
    missing_graphs = []

    for patient in split_df['patient_norm']:
        graph_path = shuffled_graph_dir / f'{patient}.pt'

        if not graph_path.exists():
            missing_graphs.append(str(graph_path))

    if missing_graphs:
        raise RuntimeError(f'Fold {fold_number} is missing {len(missing_graphs)} shuffled graphs. Examples: {missing_graphs[:10]}')

    return (train_df, val_df, test_df, split_df)


def load_labels_from_fused(fused_rows_path, case_map_path):
    rows = []

    with open(fused_rows_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')

            if len(parts) == 2:
                rows.append(parts)

    if not rows:
        raise RuntimeError(f'No valid rows found in {fused_rows_path}')
    col0 = [r[0] for r in rows]

    col1 = [r[1] for r in rows]

    tumor_col = 0 if len(set(col0)) < len(set(col1)) else 1
    patient_col = 1 - tumor_col
    fused_df = pd.DataFrame({'case_id': [r[patient_col] for r in rows], 'label_text': [r[tumor_col] for r in rows]})

    case_map = pd.read_csv(case_map_path)

    required = {'case_id', 'barcode'}

    missing_cols = required - set(case_map.columns)

    if missing_cols:
        raise RuntimeError(f'Missing columns in case map: {sorted(missing_cols)}')
    case_map = case_map[['case_id', 'barcode']].copy()

    case_map['case_id'] = case_map['case_id'].astype(str)

    fused_df['case_id'] = fused_df['case_id'].astype(str)

    df = fused_df.merge(case_map, on='case_id', how='left', validate='one_to_one')

    unmapped = df['barcode'].isna()

    if unmapped.any():
        print('WARNING: fused cases without TCGA barcode:', int(unmapped.sum()))
    df = df.loc[~unmapped].copy()

    df['patient_norm'] = df['barcode'].map(normalize_patient_id)

    df['site'] = df['patient_norm'].map(extract_tcga_site)

    if df['patient_norm'].duplicated().any():
        dup = df.loc[df['patient_norm'].duplicated(keep=False), 'patient_norm'].tolist()

        raise RuntimeError(f'Duplicate TCGA patients after mapping. Examples: {dup[:10]}')
    print('Fused rows:', len(fused_df))

    print('Mapped TCGA labels:', len(df))

    print('Classes:', df['label_text'].nunique())

    return df


def load_labels_from_csv(labels_csv_path):
    labels_csv_path = Path(labels_csv_path)

    if not labels_csv_path.exists():
        raise FileNotFoundError(f'Labels CSV not found: {labels_csv_path}. Run build_all_9547_tcga_labels.py first.')
    df = pd.read_csv(labels_csv_path)

    required = {'patient_norm', 'label_text'}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f'Missing required columns in labels CSV: {sorted(missing)}')
    df = df.copy()

    df['patient_norm'] = df['patient_norm'].map(normalize_patient_id)

    if 'site' not in df.columns:
        df['site'] = df['patient_norm'].map(extract_tcga_site)

    if df['patient_norm'].duplicated().any():
        dup = df.loc[df['patient_norm'].duplicated(keep=False), 'patient_norm'].tolist()

        raise RuntimeError(f'Duplicate patients in labels CSV. Examples: {dup[:10]}')

    if df['label_text'].isna().any():
        raise RuntimeError('Labels CSV contains missing tumor-origin labels.')
    print('Label CSV rows:', len(df))

    print('Classes:', df['label_text'].nunique())

    return df


def graph_patient_from_path(path):
    return normalize_patient_id(path.stem)


def discover_graphs(graph_dir):
    paths = sorted(Path(graph_dir).glob('*.pt'))

    if not paths:
        raise RuntimeError(f'No graph files found in {graph_dir}')
    mapping = {}

    for path in paths:
        pid = graph_patient_from_path(path)

        if pid in mapping:
            raise RuntimeError(f'Duplicate graph patient ID: {pid}')
        mapping[pid] = path

    return mapping


def resolve_h5_path(graph):
    keys = ['source_h5', 'h5_path', 'source_h5_path', 'embedding_path', 'feature_path']

    for key in keys:
        if key in graph:
            value = graph[key]

            if isinstance(value, Path):
                value = str(value)

            if isinstance(value, str):
                return value

    raise KeyError('Could not find H5 path in graph.')


def load_feature_matrix(h5_path, node_indices=None):
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise KeyError(f"'features' missing from {h5_path}")
        ds = f['features']

        if node_indices is not None:
            node_indices = np.asarray(node_indices, dtype=np.int64)

            order = np.argsort(node_indices)

            sorted_idx = node_indices[order]

        if ds.ndim == 3:
            if ds.shape[0] != 1:
                raise RuntimeError(f'Unexpected feature shape {ds.shape}')

            if node_indices is None:
                x = ds[0, :, :]
            else:
                sorted_x = ds[0, sorted_idx, :]

                reverse = np.empty_like(order)

                reverse[order] = np.arange(len(order))

                x = sorted_x[reverse]
        elif ds.ndim == 2:
            if node_indices is None:
                x = ds[:, :]
            else:
                sorted_x = ds[sorted_idx, :]

                reverse = np.empty_like(order)

                reverse[order] = np.arange(len(order))

                x = sorted_x[reverse]
        else:
            raise RuntimeError(f'Unexpected feature shape {ds.shape}')

    return np.asarray(x, dtype=np.float32)


def get_graph_components(graph):
    edge_index = graph['edge_index']

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()

    if edge_index.ndim == 2 and edge_index.shape[0] != 2 and (edge_index.shape[1] == 2):
        edge_index = edge_index.t().contiguous()

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(f'Invalid edge_index shape: {edge_index.shape}')
    edge_attr = graph.get('edge_attr')

    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()
    node_indices = graph.get('node_indices')

    if node_indices is not None:
        if torch.is_tensor(node_indices):
            node_indices = node_indices.cpu().numpy()
        node_indices = np.asarray(node_indices, dtype=np.int64)

    return (edge_index, edge_attr, node_indices)


class SpatialGraphDataset(Dataset):


    def __init__(self, rows, label_encoder):
        self.rows = rows.reset_index(drop=True)

        self.label_encoder = label_encoder


    def __len__(self):
        return len(self.rows)


    def __getitem__(self, idx):
        row = self.rows.iloc[idx]

        graph_path = Path(row['graph_path'])

        graph = torch.load(graph_path, map_location='cpu', weights_only=False)

        edge_index, edge_attr, node_indices = get_graph_components(graph)

        h5_path = resolve_h5_path(graph)

        x = load_feature_matrix(h5_path, node_indices=node_indices)

        x = torch.from_numpy(x)

        if edge_index.numel() > 0 and int(edge_index.max()) >= x.shape[0]:
            raise RuntimeError(f'edge_index exceeds node count in {graph_path.name}')
        y = int(self.label_encoder.transform([row['label_text']])[0])

        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'patient_id': row['patient_norm']}


def collate_graphs(batch):
    xs = []

    edge_indices = []

    edge_attrs = []

    batch_ids = []

    ys = []

    patient_ids = []

    node_offset = 0

    for graph_id, item in enumerate(batch):
        x = item['x']

        edge_index = item['edge_index']

        edge_attr = item['edge_attr']

        xs.append(x)

        if edge_index.numel() > 0:
            edge_indices.append(edge_index + node_offset)

            edge_attrs.append(edge_attr)
        batch_ids.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))

        ys.append(item['y'])

        patient_ids.append(item['patient_id'])

        node_offset += x.shape[0]
    x = torch.cat(xs, dim=0)

    batch_tensor = torch.cat(batch_ids, dim=0)

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)

        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_attr = torch.zeros((0, 3), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)

    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'batch': batch_tensor, 'y': y, 'patient_ids': patient_ids}


class LowRankFourierKAN(nn.Module):


    def __init__(self, input_dim, output_dim, rank=16, num_frequencies=4, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.num_frequencies = num_frequencies
        self.pre = nn.Linear(input_dim, rank)

        self.sin_coeff = nn.Parameter(torch.empty(rank, rank, num_frequencies))

        self.cos_coeff = nn.Parameter(torch.empty(rank, rank, num_frequencies))

        self.bias = nn.Parameter(torch.zeros(rank))

        self.post = nn.Linear(rank, output_dim)

        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.sin_coeff, mean=0.0, std=0.02)

        nn.init.normal_(self.cos_coeff, mean=0.0, std=0.02)


    def forward(self, x):
        z = torch.tanh(self.pre(x))

        frequencies = torch.arange(1, self.num_frequencies + 1, device=z.device, dtype=z.dtype)

        phase = z.unsqueeze(-1) * frequencies
        sin_basis = torch.sin(phase)

        cos_basis = torch.cos(phase)

        sin_term = torch.einsum('nif,oif->no', sin_basis, self.sin_coeff.to(dtype=z.dtype))

        cos_term = torch.einsum('nif,oif->no', cos_basis, self.cos_coeff.to(dtype=z.dtype))

        latent = sin_term + cos_term + self.bias.to(dtype=z.dtype)

        latent = self.dropout(latent)

        return self.post(latent)


class KANAttentionScorer(nn.Module):


    def __init__(self, num_heads, num_inputs=4, grid_size=8, grid_min=-1.0, grid_max=1.0):
        super().__init__()

        self.num_heads = num_heads
        self.num_inputs = num_inputs
        self.grid_size = grid_size
        self.grid_min = float(grid_min)

        self.grid_max = float(grid_max)

        self.coefficients = nn.Parameter(torch.empty(num_heads, num_inputs, grid_size))

        self.bias = nn.Parameter(torch.zeros(num_heads))

        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)


    def forward(self, z):
        z = z.clamp(self.grid_min, self.grid_max)

        scaled = (z - self.grid_min) / (self.grid_max - self.grid_min) * (self.grid_size - 1)

        left = torch.floor(scaled).long().clamp(0, self.grid_size - 2)

        right = left + 1
        frac = scaled - left.to(scaled.dtype)

        head_index = torch.arange(self.num_heads, device=z.device).view(1, self.num_heads, 1).expand_as(left)

        input_index = torch.arange(self.num_inputs, device=z.device).view(1, 1, self.num_inputs).expand_as(left)

        left_values = self.coefficients[head_index, input_index, left]

        right_values = self.coefficients[head_index, input_index, right]

        spline_values = left_values * (1.0 - frac) + right_values * frac

        return spline_values.sum(dim=-1) + self.bias


def segment_softmax(scores, dst, num_nodes):
    num_heads = scores.shape[1]

    dst_index = dst.view(-1, 1).expand(-1, num_heads)

    max_per_node = torch.full((num_nodes, num_heads), -torch.inf, device=scores.device, dtype=scores.dtype)

    max_per_node.scatter_reduce_(0, dst_index, scores, reduce='amax', include_self=True)

    exp_scores = torch.exp(scores - max_per_node[dst])

    denom = torch.zeros((num_nodes, num_heads), device=scores.device, dtype=scores.dtype)

    denom.scatter_add_(0, dst_index, exp_scores)

    return exp_scores / denom[dst].clamp(min=1e-12)


class SpatialKANAttention(nn.Module):


    def __init__(self, hidden_dim, edge_dim=3, num_heads=4, dropout=0.1, kan_grid_size=8, radius_multiplier=1.5, kan_rank=16, kan_frequencies=4):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f'hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}')

        if edge_dim != 3:
            raise ValueError('SpatialKANAttention expects exactly 3 edge attributes.')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.radius_multiplier = float(radius_multiplier)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.kan_scorer = KANAttentionScorer(num_heads=num_heads, num_inputs=4, grid_size=kan_grid_size, grid_min=-1.0, grid_max=1.0)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.kan_update = LowRankFourierKAN(input_dim=hidden_dim * 2, output_dim=hidden_dim, rank=kan_rank, num_frequencies=kan_frequencies, dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x
        src = edge_index[0]

        dst = edge_index[1]

        num_nodes = x.shape[0]

        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        similarity = (q[dst] * k[src]).sum(dim=-1) / self.head_dim ** 0.5
        similarity = torch.tanh(similarity)

        edge_attr_local = edge_attr.to(dtype=x.dtype)

        distance = edge_attr_local[:, 0]

        dx = edge_attr_local[:, 1]

        dy = edge_attr_local[:, 2]

        distance_norm = (2.0 * (distance / self.radius_multiplier) - 1.0).clamp(-1.0, 1.0)

        dx_norm = (dx / self.radius_multiplier).clamp(-1.0, 1.0)

        dy_norm = (dy / self.radius_multiplier).clamp(-1.0, 1.0)

        spatial_features = torch.stack([distance_norm, dx_norm, dy_norm], dim=-1).unsqueeze(1).expand(-1, self.num_heads, -1)

        attention_features = torch.cat([similarity.unsqueeze(-1), spatial_features], dim=-1)

        attention_logits = self.kan_scorer(attention_features)

        attention = segment_softmax(attention_logits, dst, num_nodes)

        attention = self.attn_dropout(attention)

        messages = (v[src] * attention.unsqueeze(-1)).to(dtype=x.dtype)

        aggregated = torch.zeros((num_nodes, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)

        aggregated.index_add_(0, dst, messages)

        aggregated = aggregated.reshape(num_nodes, self.hidden_dim)

        aggregated = self.out_proj(aggregated)

        updated = self.kan_update(torch.cat([x, aggregated], dim=1))

        return self.norm(x + updated)


def global_mean_max_pool(x, batch, num_graphs):
    dim = x.shape[1]

    mean_pool = torch.zeros((num_graphs, dim), device=x.device, dtype=x.dtype)

    mean_pool.index_add_(0, batch, x)

    counts = torch.bincount(batch, minlength=num_graphs).to(x.dtype)

    mean_pool = mean_pool / counts.clamp(min=1).unsqueeze(1)

    max_pool = torch.zeros((num_graphs, dim), device=x.device, dtype=x.dtype)

    for graph_id in range(num_graphs):
        mask = batch == graph_id

        if mask.any():
            max_pool[graph_id] = x[mask].max(dim=0).values

    return torch.cat([mean_pool, max_pool], dim=1)


class KANAttentionPool(nn.Module):


    def __init__(self, hidden_dim, kan_rank=16, kan_frequencies=4, dropout=0.1):
        super().__init__()

        self.score = LowRankFourierKAN(input_dim=hidden_dim, output_dim=1, rank=kan_rank, num_frequencies=kan_frequencies, dropout=dropout)


    def forward(self, x, batch, num_graphs):
        logits = self.score(x).squeeze(-1)

        weights = torch.zeros_like(logits)

        for graph_id in range(num_graphs):
            mask = batch == graph_id

            if mask.any():
                local_weights = torch.softmax(logits[mask], dim=0)

                weights[mask] = local_weights.to(dtype=weights.dtype)
        pooled = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=x.dtype)

        pooled.index_add_(0, batch, x * weights.unsqueeze(-1))

        return (pooled, weights)


class SpatialGATKAN(nn.Module):


    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.1, num_heads=4, kan_grid_size=8, radius_multiplier=1.5, kan_rank=16, kan_frequencies=4):
        super().__init__()

        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        self.layers = nn.ModuleList([SpatialKANAttention(hidden_dim=hidden_dim, edge_dim=3, num_heads=num_heads, dropout=dropout, kan_grid_size=kan_grid_size, radius_multiplier=radius_multiplier, kan_rank=kan_rank, kan_frequencies=kan_frequencies) for _ in range(num_layers)])

        self.attention_pool = KANAttentionPool(hidden_dim=hidden_dim, kan_rank=kan_rank, kan_frequencies=kan_frequencies, dropout=dropout)

        self.graph_readout = LowRankFourierKAN(input_dim=hidden_dim * 2, output_dim=hidden_dim, rank=kan_rank, num_frequencies=kan_frequencies, dropout=dropout)

        self.readout_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        num_graphs = int(batch.max().item()) + 1
        attention_pooled, _ = self.attention_pool(x, batch, num_graphs)

        mean_pooled = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=x.dtype)

        mean_pooled.index_add_(0, batch, x)

        counts = torch.bincount(batch, minlength=num_graphs).to(x.dtype)

        mean_pooled = mean_pooled / counts.clamp(min=1).unsqueeze(1)

        graph_state = torch.cat([attention_pooled, mean_pooled], dim=1)

        graph_state = self.graph_readout(graph_state)

        graph_state = self.readout_norm(graph_state)

        return self.classifier(graph_state)


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))

    return torch.tensor(weights, dtype=torch.float32)


def move_batch(batch, device):
    return {'x': batch['x'].to(device, non_blocking=True), 'edge_index': batch['edge_index'].to(device, non_blocking=True), 'edge_attr': batch['edge_attr'].to(device, non_blocking=True), 'batch': batch['batch'].to(device, non_blocking=True), 'y': batch['y'].to(device, non_blocking=True), 'patient_ids': batch['patient_ids']}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, num_classes):
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_y = []

    all_prob = []

    pbar = tqdm(loader, desc='train', leave=False)

    for batch in pbar:
        batch = move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])

            loss = criterion(logits, batch['y'])
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        probs = torch.softmax(logits.detach().float(), dim=1)

        bs = batch['y'].shape[0]

        total_loss += loss.item() * bs
        total_samples += bs
        all_y.extend(batch['y'].detach().cpu().numpy().tolist())

        all_prob.append(probs.cpu().numpy())

        pbar.set_postfix(loss=f'{loss.item():.4f}')
    y_true = np.asarray(all_y, dtype=np.int64)

    y_prob = np.concatenate(all_prob, axis=0)

    y_pred = y_prob.argmax(axis=1)

    top1 = accuracy_score(y_true, y_pred)

    top3 = top_k_accuracy_score(y_true, y_prob, k=min(3, num_classes), labels=np.arange(num_classes))

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {'loss': total_loss / max(total_samples, 1), 'top1': float(top1), 'top3': float(top3), 'weighted_f1': float(weighted_f1)}

@torch.no_grad()


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_y = []

    all_prob = []

    all_patient_ids = []

    for batch in tqdm(loader, desc='eval', leave=False):
        batch = move_batch(batch, device)

        logits = model(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])

        loss = criterion(logits, batch['y'])

        probs = torch.softmax(logits, dim=1)

        bs = batch['y'].shape[0]

        total_loss += loss.item() * bs
        total_samples += bs
        all_y.extend(batch['y'].cpu().numpy().tolist())

        all_prob.append(probs.cpu().numpy())

        all_patient_ids.extend(batch['patient_ids'])
    y_true = np.asarray(all_y, dtype=np.int64)

    y_prob = np.concatenate(all_prob, axis=0)

    y_pred = y_prob.argmax(axis=1)

    top1 = accuracy_score(y_true, y_pred)

    top3 = top_k_accuracy_score(y_true, y_prob, k=min(3, num_classes), labels=np.arange(num_classes))

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {'loss': total_loss / max(total_samples, 1), 'top1': float(top1), 'top3': float(top3), 'weighted_f1': float(weighted_f1), 'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob, 'patient_ids': all_patient_ids}


def safe_probability_column_name(label):
    cleaned = str(label).strip().lower()

    cleaned = re.sub('[^a-z0-9]+', '_', cleaned).strip('_')

    return f'prob_{cleaned}'


def save_predictions(result, label_encoder, path):
    rows = []

    class_names = label_encoder.classes_.tolist()

    probability_columns = [safe_probability_column_name(label) for label in class_names]

    for i, patient_id in enumerate(result['patient_ids']):
        prob = result['y_prob'][i]

        top3_idx = np.argsort(prob)[::-1][:3]

        row = {'patient_id': patient_id, 'true_label': label_encoder.inverse_transform([int(result['y_true'][i])])[0], 'pred_label': label_encoder.inverse_transform([int(result['y_pred'][i])])[0]}

        for rank, class_idx in enumerate(top3_idx, start=1):
            row[f'top{rank}_label'] = label_encoder.inverse_transform([int(class_idx)])[0]

            row[f'top{rank}_prob'] = float(prob[class_idx])

        for class_idx, column_name in enumerate(probability_columns):
            row[column_name] = float(prob[class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_evaluation_artifacts(result, label_encoder, output_dir, prefix):
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(result, label_encoder, output_dir / f'{prefix}_predictions.csv')

    class_names = label_encoder.classes_.tolist()

    cm = confusion_matrix(result['y_true'], result['y_pred'], labels=np.arange(len(class_names)))

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f'{prefix}_confusion_matrix.csv')

    report = classification_report(result['y_true'], result['y_pred'], labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)

    pd.DataFrame(report).transpose().to_csv(output_dir / f'{prefix}_classification_report.csv')

    np.savez_compressed(output_dir / f'{prefix}_raw_outputs.npz', y_true=result['y_true'], y_pred=result['y_pred'], y_prob=result['y_prob'], patient_ids=np.asarray(result['patient_ids'], dtype=object), class_names=np.asarray(class_names, dtype=object))
BLUE_CM_COLORS = ['#E8F4F8', '#B8D9E8', '#88BED8', '#5AA3C8', '#3D88B8', '#2A6D9D', '#1e5c8e', '#143D5E', '#0A1E2E']
CURVE_COLORS = ['#1f77b4', '#9467bd', '#2ca02c', '#d62728', '#ff7f0e']


def save_combined_five_fold_curves(output_dir, n_folds=5, condition='Spatial-Null'):
    output_dir = Path(output_dir)

    histories = []

    for fold in range(1, n_folds + 1):
        history_path = output_dir / f'fold_{fold}' / 'training_history.csv'

        if not history_path.exists():
            return
        history_df = pd.read_csv(history_path)

        if 'epoch' not in history_df.columns:
            return
        histories.append((fold, history_df))
    plot_specs = [('train_loss', 'Training Loss', 'Training Loss per Fold', 'training_loss_5fold.png'), ('train_top1_accuracy', 'Training Accuracy', 'Training Accuracy per Fold', 'training_accuracy_5fold.png'), ('val_loss', 'Validation Loss', 'Validation Loss per Fold', 'validation_loss_5fold.png'), ('val_top1_accuracy', 'Validation Accuracy', 'Validation Accuracy per Fold', 'validation_accuracy_5fold.png')]

    for column, ylabel, title, filename in plot_specs:
        if any((column not in df.columns for _, df in histories)):
            return
        fig, ax = plt.subplots(figsize=(10, 6))

        for fold, history_df in histories:
            ax.plot(history_df['epoch'], history_df[column], color=CURVE_COLORS[(fold - 1) % len(CURVE_COLORS)], linewidth=3.2, alpha=0.95, label=f'Fold {fold}')
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=12)

        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=16)

        ax.set_title(f'{condition} | {title}', fontsize=30, fontweight='bold', pad=20)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        legend = ax.legend(fontsize=15, frameon=True, shadow=True)

        for label in legend.get_texts():
            label.set_fontweight('bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        ax.tick_params(axis='both', labelsize=12)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        fig.tight_layout()

        fig.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')

        plt.close(fig)


def save_combined_confusion_matrix(output_dir, n_folds=5):
    output_dir = Path(output_dir)

    combined = None

    for fold in range(1, n_folds + 1):
        path = output_dir / f'fold_{fold}' / 'test_confusion_matrix.csv'

        if not path.exists():
            return
        matrix = pd.read_csv(path, index_col=0)

        combined = matrix.copy() if combined is None else combined.add(matrix, fill_value=0)
    combined = combined.fillna(0).astype(int)

    combined.to_csv(output_dir / 'combined_confusion_matrix.csv')

    values = combined.values
    positive = values[values > 0]

    if positive.size == 0:
        return
    cmap = LinearSegmentedColormap.from_list('custom_blues', BLUE_CM_COLORS)

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(values, interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=max(float(positive.min()), 0.5), vmax=max(float(values.max()), 1.0)))

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    colorbar.ax.tick_params(labelsize=11)

    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    class_names = combined.index.tolist()

    ticks = list(range(len(class_names)))

    ax.set_xticks(ticks)

    ax.set_yticks(ticks)

    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=13, fontweight='bold')

    ax.set_yticklabels(class_names, fontsize=13, fontweight='bold')

    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=12)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=12)

    ax.set_title('Confusion Matrix', fontsize=25, fontweight='bold', pad=20)

    fig.tight_layout()

    fig.savefig(output_dir / 'combined_confusion_matrix.png', dpi=600, bbox_inches='tight', facecolor='white')

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_dir', default=DEFAULT_GRAPH_DIR)

    parser.add_argument('--fused_rows', default=DEFAULT_FUSED_ROWS)

    parser.add_argument('--labels_csv', default=DEFAULT_LABELS_CSV, help='CSV with patient_norm,label_text[,site,project_id]. Default is the all-9547 TCGA label file.')

    parser.add_argument('--case_map', default=DEFAULT_CASE_MAP)

    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT)

    parser.add_argument('--split_dir', default=DEFAULT_SPLIT_DIR, help='Directory containing the pre-generated fold_1 ... fold_5 split.csv files.')

    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--kan_grid_size', type=int, default=8)

    parser.add_argument('--radius_multiplier', type=float, default=1.5)

    parser.add_argument('--kan_rank', type=int, default=16)

    parser.add_argument('--kan_frequencies', type=int, default=4)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--weight_decay', type=float, default=1e-05)

    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--n_splits', type=int, default=5)

    parser.add_argument('--only_fold', type=int, default=None, help='Run only one outer fold using 1-based numbering, for example --only_fold 5. If omitted, run all folds.')

    parser.add_argument('--inner_val_splits', type=int, default=8)

    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.labels_csv:
        labels_df = load_labels_from_csv(args.labels_csv)
    else:
        labels_df = load_labels_from_fused(args.fused_rows, args.case_map)
    graph_map = discover_graphs(args.graph_dir)

    print('Graphs discovered:', f'{len(graph_map):,}')

    labels_df['graph_path'] = labels_df['patient_norm'].map(lambda x: str(graph_map[x]) if x in graph_map else None)

    missing = labels_df['graph_path'].isna()

    print('Labels without graph:', int(missing.sum()))

    df = labels_df.loc[~missing].copy().reset_index(drop=True)

    print('Usable patients:', f'{len(df):,}')

    print('Classes:', df['label_text'].nunique())

    print('TCGA sites:', df['site'].nunique())

    print(df['label_text'].value_counts().to_string())

    predefined_outer_path = Path(args.split_dir) / 'site_stratified_outer_folds.csv'

    if not predefined_outer_path.exists():
        raise FileNotFoundError(f'Missing pre-generated outer fold file: {predefined_outer_path}')
    predefined_outer = pd.read_csv(predefined_outer_path, keep_default_na=False)

    if len(predefined_outer) != len(df):
        raise RuntimeError(f'Pre-generated outer-fold file has {len(predefined_outer):,} rows but the current pathology cohort has {len(df):,} patients.')
    print('Split directory:', args.split_dir)

    print('Outer-fold file:', predefined_outer_path)

    predefined_outer.to_csv(output_dir / 'site_stratified_outer_folds.csv', index=False)

    label_encoder = LabelEncoder()

    label_encoder.fit(df['label_text'])

    num_classes = len(label_encoder.classes_)

    with open(output_dir / 'label_classes.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f, indent=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:', device)

    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    fold_metrics = []

    all_oof_rows = []

    if args.only_fold is not None:
        if args.only_fold < 1 or args.only_fold > args.n_splits:
            raise ValueError(f'--only_fold must be between 1 and {args.n_splits}, got {args.only_fold}')
        folds_to_run = [args.only_fold - 1]

        print(f'Running ONLY outer fold {args.only_fold}/{args.n_splits}')
    else:
        folds_to_run = list(range(args.n_splits))

    for fold in folds_to_run:
        print(f'OUTER FOLD {fold + 1}/{args.n_splits}')

        fold_dir = output_dir / f'fold_{fold + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df, val_df, test_df, predefined_split_df = load_predefined_split(split_dir=args.split_dir, fold_number=fold + 1)

        expected_total = len(df)

        actual_total = len(train_df) + len(val_df) + len(test_df)

        if actual_total != expected_total:
            raise RuntimeError(f'Fold {fold + 1} split contains {actual_total:,} patients; expected {expected_total:,}.')
        print('Train:', len(train_df))

        print('Validation:', len(val_df))

        print('Test:', len(test_df))

        split_rows = pd.concat([train_df.assign(split='train'), val_df.assign(split='val'), test_df.assign(split='test')], ignore_index=True)

        split_rows[['patient_norm', 'label_text', 'site', 'graph_path', 'split']].to_csv(fold_dir / 'split.csv', index=False)

        train_ds = SpatialGraphDataset(train_df, label_encoder)

        val_ds = SpatialGraphDataset(val_df, label_encoder)

        test_ds = SpatialGraphDataset(test_df, label_encoder)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_graphs)

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_graphs)

        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_graphs)

        sample = train_ds[0]

        input_dim = sample['x'].shape[1]

        print('Input embedding dim:', input_dim)

        model = SpatialGATKAN(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.num_layers, dropout=args.dropout, num_heads=args.num_heads, kan_grid_size=args.kan_grid_size, radius_multiplier=args.radius_multiplier, kan_rank=args.kan_rank, kan_frequencies=args.kan_frequencies).to(device)

        train_labels_encoded = label_encoder.transform(train_df['label_text'])

        class_weights = compute_class_weights(train_labels_encoded, num_classes).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-07)

        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_top1 = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        best_path = fold_dir / 'best_model.pt'
        history = []

        for epoch in range(1, args.epochs + 1):
            print(f'Fold {fold + 1} | Epoch {epoch}/{args.epochs}')

            train_result = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler, device=device, use_amp=use_amp, num_classes=num_classes)

            val_result = evaluate(model=model, loader=val_loader, criterion=criterion, device=device, num_classes=num_classes)

            scheduler.step(val_result['top1'])

            lr_now = optimizer.param_groups[0]['lr']

            history_row = {'epoch': epoch, 'train_loss': train_result['loss'], 'train_top1_accuracy': train_result['top1'], 'train_top3_accuracy': train_result['top3'], 'train_weighted_f1': train_result['weighted_f1'], 'val_loss': val_result['loss'], 'val_top1_accuracy': val_result['top1'], 'val_top3_accuracy': val_result['top3'], 'val_weighted_f1': val_result['weighted_f1'], 'lr': lr_now}

            history.append(history_row)

            pd.DataFrame(history).to_csv(fold_dir / 'training_history.csv', index=False)

            print(f"train_loss={train_result['loss']:.4f} | train_acc={train_result['top1']:.4f} | val_loss={val_result['loss']:.4f} | val_acc={val_result['top1']:.4f} | val_top3={val_result['top3']:.4f} | val_f1={val_result['weighted_f1']:.4f} | lr={lr_now:.2e}")

            if val_result['top1'] > best_top1:
                best_top1 = val_result['top1']

                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'best_val_top1': best_top1, 'input_dim': input_dim, 'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'kan_grid_size': args.kan_grid_size, 'radius_multiplier': args.radius_multiplier, 'kan_rank': args.kan_rank, 'kan_frequencies': args.kan_frequencies, 'dropout': args.dropout, 'architecture': 'SpatialKAGAT_FourierKAN', 'classes': label_encoder.classes_.tolist()}, best_path)

                save_evaluation_artifacts(val_result, label_encoder, fold_dir, 'best_validation')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f'Early stopping after {args.patience} epochs without validation Top-1 improvement.')

                break
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        print(f'TESTING OUTER FOLD {fold + 1} BEST MODEL FROM EPOCH {best_epoch}')

        test_result = evaluate(model=model, loader=test_loader, criterion=criterion, device=device, num_classes=num_classes)

        fold_result = {'fold': fold + 1, 'best_epoch': best_epoch, 'best_val_top1': float(checkpoint['best_val_top1']), 'test_loss': test_result['loss'], 'test_top1': test_result['top1'], 'test_top3': test_result['top3'], 'test_weighted_f1': test_result['weighted_f1'], 'n_train': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df)}

        fold_metrics.append(fold_result)

        save_evaluation_artifacts(test_result, label_encoder, fold_dir, 'test')

        with open(fold_dir / 'test_metrics.json', 'w') as f:
            json.dump(fold_result, f, indent=2)
        test_probs = test_result['y_prob']

        for i, patient_id in enumerate(test_result['patient_ids']):
            row = {'patient_id': patient_id, 'fold': fold + 1, 'true_label': label_encoder.inverse_transform([int(test_result['y_true'][i])])[0], 'pred_label': label_encoder.inverse_transform([int(test_result['y_pred'][i])])[0]}

            top3_idx = np.argsort(test_probs[i])[::-1][:3]

            for rank, class_idx in enumerate(top3_idx, start=1):
                row[f'top{rank}_label'] = label_encoder.inverse_transform([int(class_idx)])[0]

                row[f'top{rank}_prob'] = float(test_probs[i, class_idx])
            all_oof_rows.append(row)
        print(f"Fold {fold + 1} Test Top-1: {test_result['top1']:.4f}")

        print(f"Fold {fold + 1} Test Top-3: {test_result['top3']:.4f}")

        print(f"Fold {fold + 1} Test Weighted F1: {test_result['weighted_f1']:.4f}")

        del model
        del optimizer
        del scaler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics_df = pd.DataFrame(fold_metrics)

    if args.only_fold is not None:
        selected_fold = int(args.only_fold)

        fold_dir = output_dir / f'fold_{selected_fold}'
        metrics_df.to_csv(fold_dir / 'fold_run_metrics.csv', index=False)

        with open(fold_dir / 'fold_run_complete.json', 'w') as f:
            json.dump({'fold': selected_fold, 'completed': True, 'metrics': fold_metrics[0] if fold_metrics else None}, f, indent=2)
        print(f'OUTER FOLD {selected_fold} COMPLETE')

        if fold_metrics:
            result = fold_metrics[0]

            print('Top-1:', f"{result['test_top1']:.4f}")

            print('Top-3:', f"{result['test_top3']:.4f}")

            print('Weighted F1:', f"{result['test_weighted_f1']:.4f}")
        print('Results saved to:', fold_dir)
    else:
        metrics_df.to_csv(output_dir / 'cross_validation_fold_metrics.csv', index=False)

        pd.DataFrame(all_oof_rows).to_csv(output_dir / 'out_of_fold_predictions.csv', index=False)

        summary = {'n_splits': args.n_splits, 'n_patients': len(df), 'site_definition': 'TCGA Tissue Source Site code from patient barcode', 'site_split_policy': 'sites stratified across folds; sites are not held out', 'mean_test_top1': float(metrics_df['test_top1'].mean()), 'std_test_top1': float(metrics_df['test_top1'].std(ddof=1)), 'mean_test_top3': float(metrics_df['test_top3'].mean()), 'std_test_top3': float(metrics_df['test_top3'].std(ddof=1)), 'mean_test_weighted_f1': float(metrics_df['test_weighted_f1'].mean()), 'std_test_weighted_f1': float(metrics_df['test_weighted_f1'].std(ddof=1))}

        with open(output_dir / 'cross_validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        with open(output_dir / 'experiment_config.json', 'w') as f:
            json.dump({'architecture': 'SpatialKAGAT_FourierKAN', 'graph_dir': str(args.graph_dir), 'n_splits': args.n_splits, 'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'kan_grid_size': args.kan_grid_size, 'kan_rank': args.kan_rank, 'kan_frequencies': args.kan_frequencies, 'radius_multiplier': args.radius_multiplier, 'dropout': args.dropout, 'learning_rate': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'epochs': args.epochs, 'patience': args.patience, 'seed': args.seed, 'site_definition': 'TCGA Tissue Source Site code from patient barcode'}, f, indent=2)
        print('Top-1:', f"{summary['mean_test_top1']:.4f} +/- {summary['std_test_top1']:.4f}")

        print('Top-3:', f"{summary['mean_test_top3']:.4f} +/- {summary['std_test_top3']:.4f}")

        print('Weighted F1:', f"{summary['mean_test_weighted_f1']:.4f} +/- {summary['std_test_weighted_f1']:.4f}")

        print('Results saved to:', output_dir)

    if args.only_fold is None:
        save_combined_five_fold_curves(output_dir=output_dir, n_folds=args.n_splits)

        save_combined_confusion_matrix(output_dir=output_dir, n_folds=args.n_splits)


if __name__ == '__main__':
    main()
