#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_GRAPH_DIR = '/workspace/data/tcga_spatial_graphs'
DEFAULT_LABELS_CSV = '/workspace/metadata/tcga_labels.csv'
DEFAULT_OUTPUT_DIR = '/workspace/splits'


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


def discover_graphs(graph_dir):
    graph_dir = Path(graph_dir)

    graph_paths = sorted(graph_dir.glob('*.pt'))

    if not graph_paths:
        raise RuntimeError(f'No graph files found in {graph_dir}')
    graph_map = {}

    for path in graph_paths:
        patient = normalize_patient_id(path.name)

        if patient in graph_map:
            raise RuntimeError(f'Duplicate graph patient ID: {patient}')
        graph_map[patient] = path

    return graph_map


def load_labels(labels_csv):
    labels_csv = Path(labels_csv)

    if not labels_csv.exists():
        raise FileNotFoundError(f'Labels CSV not found: {labels_csv}\nRun the all-9547 TCGA label-building program first.')
    df = pd.read_csv(labels_csv, keep_default_na=False)

    required = {'patient_norm', 'label_text'}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f'Labels CSV is missing columns: {sorted(missing)}')
    df = df.copy()

    df['patient_norm'] = df['patient_norm'].map(normalize_patient_id)

    df['site'] = df['patient_norm'].map(extract_tcga_site)

    if df['patient_norm'].duplicated().any():
        duplicates = df.loc[df['patient_norm'].duplicated(keep=False), 'patient_norm'].drop_duplicates().tolist()

        raise RuntimeError(f'Duplicate patients in labels CSV. Examples: {duplicates[:20]}')

    if df['label_text'].isna().any():
        raise RuntimeError('Labels CSV contains missing tumor-origin labels.')

    if df['site'].isna().any():
        raise RuntimeError('Labels CSV contains missing site values.')

    return df


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
        raise RuntimeError('Fold assignment left unassigned patients.')

    return assignments


def print_outer_qc(df, n_splits):
    total_classes = df['label_text'].nunique()

    total_sites = df['site'].nunique()

    for fold in range(n_splits):
        fold_df = df.loc[df['outer_fold'] == fold]

        print(f"Fold {fold + 1}: n={len(fold_df):,} | classes={fold_df['label_text'].nunique()}/{total_classes} | sites={fold_df['site'].nunique()}/{total_sites}")


def build_train_val_test_splits(df, n_splits, inner_val_splits, seed, output_dir):
    summary_rows = []

    for fold in range(n_splits):
        test_df = df.loc[df['outer_fold'] == fold].copy()

        outer_train_df = df.loc[df['outer_fold'] != fold].copy().reset_index(drop=True)

        inner_assignments = make_site_stratified_folds(df=outer_train_df, n_splits=inner_val_splits, seed=seed + 1000 + fold)

        outer_train_df['inner_fold'] = inner_assignments
        val_df = outer_train_df.loc[outer_train_df['inner_fold'] == 0].copy()

        train_df = outer_train_df.loc[outer_train_df['inner_fold'] != 0].copy()

        fold_dir = output_dir / f'fold_{fold + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        split_df = pd.concat([train_df.assign(split='train'), val_df.assign(split='val'), test_df.assign(split='test')], ignore_index=True)

        split_df[['patient_norm', 'label_text', 'site', 'graph_path', 'split']].to_csv(fold_dir / 'split.csv', index=False)

        class_table = split_df.groupby(['label_text', 'split']).size().unstack(fill_value=0)

        class_table.to_csv(fold_dir / 'class_counts.csv')

        site_table = split_df.groupby(['site', 'split']).size().unstack(fill_value=0)

        site_table.to_csv(fold_dir / 'site_counts.csv')

        summary_rows.append({'fold': fold + 1, 'train_n': len(train_df), 'val_n': len(val_df), 'test_n': len(test_df), 'train_classes': train_df['label_text'].nunique(), 'val_classes': val_df['label_text'].nunique(), 'test_classes': test_df['label_text'].nunique(), 'train_sites': train_df['site'].nunique(), 'val_sites': val_df['site'].nunique(), 'test_sites': test_df['site'].nunique()})
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(output_dir / 'fold_summary.csv', index=False)

    return summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR)

    parser.add_argument('--labels-csv', default=DEFAULT_LABELS_CSV)

    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--n-splits', type=int, default=5)

    parser.add_argument('--inner-val-splits', type=int, default=8, help='Validation is one of 8 folds from the outer 80%% pool, giving approximately 10%% of the full cohort for validation.')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print('Graph directory:', args.graph_dir)

    print('Labels CSV:     ', args.labels_csv)

    print('Output:         ', output_dir)

    print('Seed:           ', args.seed)

    labels_df = load_labels(args.labels_csv)

    graph_map = discover_graphs(args.graph_dir)

    print('Graph patients:', f'{len(graph_map):,}')

    print('Label patients:', f'{len(labels_df):,}')

    labels_df['graph_path'] = labels_df['patient_norm'].map(lambda patient: str(graph_map[patient]) if patient in graph_map else None)

    missing_graph = labels_df['graph_path'].isna()

    if missing_graph.any():
        missing_patients = labels_df.loc[missing_graph, 'patient_norm'].tolist()

        raise RuntimeError(f'{len(missing_patients)} labeled patients do not have graphs. Examples: {missing_patients[:20]}')
    graph_without_label = sorted(set(graph_map) - set(labels_df['patient_norm']))

    if graph_without_label:
        raise RuntimeError(f'{len(graph_without_label)} graph patients do not have labels. Examples: {graph_without_label[:20]}')
    df = labels_df.copy().reset_index(drop=True)

    if len(df) != len(graph_map):
        raise RuntimeError(f'Patient mismatch: {len(df):,} labeled vs {len(graph_map):,} graphs.')
    print('Usable patients:', f'{len(df):,}')

    print('Tumor origins: ', df['label_text'].nunique())

    print('TCGA TSS sites:', df['site'].nunique())

    outer_assignments = make_site_stratified_folds(df=df, n_splits=args.n_splits, seed=args.seed)

    df['outer_fold'] = outer_assignments
    print_outer_qc(df, args.n_splits)

    outer_csv = output_dir / 'site_stratified_outer_folds.csv'
    df[['patient_norm', 'label_text', 'site', 'graph_path', 'outer_fold']].assign(outer_fold=lambda x: x['outer_fold'] + 1).to_csv(outer_csv, index=False)

    summary_df = build_train_val_test_splits(df=df, n_splits=args.n_splits, inner_val_splits=args.inner_val_splits, seed=args.seed, output_dir=output_dir)

    class_counts = df['label_text'].value_counts().sort_index()

    class_counts.to_csv(output_dir / 'overall_class_counts.csv', header=['n'])

    metadata = {'n_patients': int(len(df)), 'n_graphs': int(len(graph_map)), 'n_classes': int(df['label_text'].nunique()), 'n_sites': int(df['site'].nunique()), 'outer_folds': int(args.n_splits), 'inner_val_splits': int(args.inner_val_splits), 'seed': int(args.seed), 'site_definition': 'TCGA Tissue Source Site (second barcode segment)', 'site_strategy': 'stratified across folds; sites are not held out as groups', 'validation_strategy': 'one of eight site-stratified inner folds from the outer training pool'}

    with open(output_dir / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(summary_df.to_string(index=False))

    print('Outer fold file:', outer_csv)

    print('Per-fold splits:', output_dir / 'fold_1/split.csv', '...', output_dir / 'fold_5/split.csv')


if __name__ == '__main__':
    main()
