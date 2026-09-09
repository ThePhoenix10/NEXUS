#!/usr/bin/env python3

import argparse
import importlib.util
import json
import re
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

DEFAULT_RESULTS_DIR = '/workspace/results/NEXUS'
DEFAULT_CPTAC_GRAPH_DIR = '/workspace/data/cptac_spatial_graphs'
DEFAULT_TRAINING_SCRIPT = '/workspace/scripts/train_spatial_aware.py'
DEFAULT_OUTPUT_DIR = '/workspace/results/cptac'
PROJECT_TO_BIOLOGICAL_LABEL = {'cptac_brca': 'Breast', 'cptac_ccrcc': 'Kidney', 'cptac_coad': 'Colon', 'cptac_gbm': 'Brain', 'cptac_hnsc': 'Head and Neck', 'cptac_lscc': 'Lung', 'cptac_luad': 'Lung', 'cptac_luad_part1': 'Lung', 'cptac_luad_part2': 'Lung', 'cptac_ov': 'Ovary', 'cptac_pda': 'Pancreas', 'cptac_ucec': 'Uterus'}
LABEL_ALIASES = {'Breast': ['Breast', 'BRCA'], 'Kidney': ['Kidney', 'Renal', 'KIRC', 'CCRCC'], 'Colon': ['Colon', 'COAD', 'Colorectal'], 'Brain': ['Brain', 'GBM', 'Glioma'], 'Head and Neck': ['Head and Neck', 'Head_Neck', 'HNSC', 'Head & Neck'], 'Lung': ['Lung', 'LUAD', 'LUSC', 'LSCC'], 'Ovary': ['Ovary', 'Ovarian', 'OV'], 'Pancreas': ['Pancreas', 'Pancreatic', 'PAAD', 'PDA'], 'Uterus': ['Uterus', 'Uterine', 'Endometrial', 'UCEC']}


def parse_args():
    parser = argparse.ArgumentParser(description='Load the five TCGA fold Spatial KA-GAT models, run each model on every CPTAC slide, average the five probability vectors, and average the five fold probability vectors and report slide-level and patient-level external validation.')

    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR, help='Directory containing fold_1 ... fold_5.')

    parser.add_argument('--cptac-graph-dir', default=DEFAULT_CPTAC_GRAPH_DIR)

    parser.add_argument('--training-script', default=DEFAULT_TRAINING_SCRIPT)


    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])

    parser.add_argument('--checkpoint-name', default='best_model.pt')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--hidden-dim', type=int, default=256)

    parser.add_argument('--num-layers', type=int, default=3)

    parser.add_argument('--num-heads', type=int, default=4)

    parser.add_argument('--kan-grid-size', type=int, default=8)

    parser.add_argument('--kan-rank', type=int, default=16)

    parser.add_argument('--kan-frequencies', type=int, default=4)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--radius-multiplier', type=float, default=1.5)

    return parser.parse_args()


def import_module_from_path(name, path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Python file not found: {path}')
    spec = importlib.util.spec_from_file_location(name, str(path))

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module




def load_graph_features(graph_path, training_module):
    import h5py

    graph = torch.load(graph_path, map_location='cpu', weights_only=False)

    edge_index, edge_attr, node_indices = training_module.get_graph_components(graph)

    source_h5 = Path(graph['source_h5'])

    if not source_h5.exists():
        raise FileNotFoundError(f'Source H5 does not exist: {source_h5}')

    with h5py.File(source_h5, 'r') as f:
        if 'features' not in f:
            raise KeyError(f"'features' missing from {source_h5}")

        features = np.asarray(f['features'])

    if features.ndim == 3:
        if features.shape[0] != 1:
            raise RuntimeError(f'Unexpected feature shape {features.shape} in {source_h5}')
        features = features[0]
    elif features.ndim != 2:
        raise RuntimeError(f'Unexpected feature shape {features.shape} in {source_h5}')

    if node_indices is not None:
        idx = node_indices.detach().cpu().numpy() if torch.is_tensor(node_indices) else np.asarray(node_indices)
        idx = idx.astype(np.int64, copy=False)
        features = features[idx]

    x = torch.as_tensor(features, dtype=torch.float32)

    return x, edge_index.long(), edge_attr.float(), source_h5, graph


def normalize_text(x):
    return re.sub('[^a-z0-9]+', '', str(x).lower())


def resolve_tcga_label(biological_label, classes):
    normalized = {normalize_text(c): c for c in classes}

    candidates = LABEL_ALIASES.get(biological_label, [biological_label])

    for candidate in candidates:
        key = normalize_text(candidate)

        if key in normalized:
            return normalized[key]

    for cls in classes:
        cls_norm = normalize_text(cls)

        for candidate in candidates:
            candidate_norm = normalize_text(candidate)

            if candidate_norm in cls_norm or cls_norm in candidate_norm:
                return cls

    raise RuntimeError(f"Could not map CPTAC label '{biological_label}' to TCGA classes: {classes}")


def detect_cptac_project(graph_path, graph):
    candidates = [str(graph_path).lower(), str(graph.get('source_h5', '')).lower()]

    for project in sorted(PROJECT_TO_BIOLOGICAL_LABEL, key=len, reverse=True):
        if any((project in value for value in candidates)):
            return project

    raise RuntimeError(f"Could not determine CPTAC project. graph={graph_path}; source_h5={graph.get('source_h5')}")


def infer_cptac_patient(source_h5):
    stem = Path(source_h5).stem
    match = re.match('^(C3[NL]-\\d+)', stem, flags=re.IGNORECASE)

    if match:
        return match.group(1).upper()
    match = re.match('^([0-9]{2}[A-Za-z]{2}\\d{3})-', stem)

    if match:
        return match.group(1).upper()

    return stem


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint is not a dictionary.')

    for key in ['model_state_dict', 'state_dict', 'model']:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if checkpoint and all((torch.is_tensor(v) for v in checkpoint.values())):
        return checkpoint

    raise RuntimeError('Could not find model weights in checkpoint.')


def get_classes(checkpoint):
    for key in ['classes', 'class_names', 'labels']:
        if key in checkpoint:
            value = checkpoint[key]

            if isinstance(value, (list, tuple)):
                return list(value)

    if 'class_to_idx' in checkpoint:
        mapping = checkpoint['class_to_idx']

        if isinstance(mapping, dict):
            return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]

    raise RuntimeError('Checkpoint does not contain class names.')


def checkpoint_value(checkpoint, key, fallback):
    value = checkpoint.get(key, fallback)

    if value is None:
        value = fallback

    return value


def build_model_from_checkpoint(training_module, checkpoint, classes, args, device):
    model = training_module.SpatialGATKAN(input_dim=int(checkpoint_value(checkpoint, 'input_dim', 1536)), hidden_dim=int(checkpoint_value(checkpoint, 'hidden_dim', args.hidden_dim)), num_classes=len(classes), num_layers=int(checkpoint_value(checkpoint, 'num_layers', args.num_layers)), dropout=float(checkpoint_value(checkpoint, 'dropout', args.dropout)), num_heads=int(checkpoint_value(checkpoint, 'num_heads', args.num_heads)), kan_grid_size=int(checkpoint_value(checkpoint, 'kan_grid_size', args.kan_grid_size)), radius_multiplier=float(checkpoint_value(checkpoint, 'radius_multiplier', args.radius_multiplier)), kan_rank=int(checkpoint_value(checkpoint, 'kan_rank', args.kan_rank)), kan_frequencies=int(checkpoint_value(checkpoint, 'kan_frequencies', args.kan_frequencies)))

    state_dict = get_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)

    model.to(device)

    model.eval()

    return model


def load_fold_models(training_module, args):
    models = []

    classes_reference = None
    fold_metadata = []

    results_dir = Path(args.results_dir)

    for fold in args.folds:
        checkpoint_path = results_dir / f'fold_{fold}' / args.checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Fold {fold} checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        classes = get_classes(checkpoint)

        if classes_reference is None:
            classes_reference = classes
        elif classes != classes_reference:
            raise RuntimeError(f'Class ordering differs in fold {fold}.\nReference: {classes_reference}\nFold {fold}: {classes}')
        model = build_model_from_checkpoint(training_module=training_module, checkpoint=checkpoint, classes=classes, args=args, device=args.device)

        models.append(model)

        fold_metadata.append({'fold': int(fold), 'checkpoint': str(checkpoint_path), 'best_epoch': checkpoint.get('best_epoch', checkpoint.get('epoch', None))})

        print(f'Fold {fold}: loaded {checkpoint_path}')
    print(f'Loaded {len(models)} fold models.')

    print(f'Classes: {len(classes_reference)}')

    return (models, classes_reference, fold_metadata)

@torch.no_grad()


def predict_probabilities(model, x, edge_index, edge_attr, device):
    x = x.to(device, non_blocking=True)

    edge_index = edge_index.to(device, non_blocking=True)

    edge_attr = edge_attr.to(device, non_blocking=True)

    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    logits = model(x, edge_index, edge_attr, batch)

    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = torch.softmax(logits, dim=1)[0]

    return probabilities.detach().float().cpu().numpy()


def safe_class_name(class_name):
    return re.sub('[^A-Za-z0-9]+', '_', class_name).strip('_')


def calculate_metrics(df):
    return {'n': int(len(df)), 'top1': float(accuracy_score(df['true_label'], df['pred_label'])), 'top3': float(df['top3_correct'].mean()), 'weighted_f1': float(f1_score(df['true_label'], df['pred_label'], average='weighted', zero_division=0))}


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    training_module = import_module_from_path('train_spatial_gnn_ensemble', args.training_script)

    models, classes, fold_metadata = load_fold_models(training_module, args)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    resolved_labels = {biological: resolve_tcga_label(biological, classes) for biological in sorted(set(PROJECT_TO_BIOLOGICAL_LABEL.values()))}

    graph_files = sorted(Path(args.cptac_graph_dir).glob('*.pt'))

    if not graph_files:
        raise RuntimeError(f'No CPTAC graphs found in {args.cptac_graph_dir}')
    print(f'Models:                       {len(models)}')

    print(f'CPTAC graphs:                 {len(graph_files)}')

    rows = []

    failures = []

    for graph_path in tqdm(graph_files, desc='Testing CPTAC', unit='slide'):
        try:
            x, edge_index, edge_attr, source_h5, graph = load_graph_features(graph_path, training_module)

            project = detect_cptac_project(graph_path, graph)

            biological_label = PROJECT_TO_BIOLOGICAL_LABEL[project]

            true_label = resolved_labels[biological_label]

            true_idx = class_to_idx[true_label]

            fold_probabilities = []

            for model in models:
                fold_probabilities.append(predict_probabilities(model=model, x=x, edge_index=edge_index, edge_attr=edge_attr, device=device))
            fold_probabilities = np.stack(fold_probabilities, axis=0)

            mean_probs = fold_probabilities.mean(axis=0)

            order = np.argsort(mean_probs)[::-1]

            pred_idx = int(order[0])

            row = {'graph': str(graph_path), 'source_h5': str(source_h5), 'project': project, 'patient_id': infer_cptac_patient(source_h5), 'true_label': true_label, 'pred_label': classes[pred_idx], 'pred_prob': float(mean_probs[pred_idx]), 'correct': int(pred_idx == true_idx), 'top3_correct': int(true_idx in order[:3]), 'top1_label': classes[order[0]], 'top1_prob': float(mean_probs[order[0]]), 'top2_label': classes[order[1]], 'top2_prob': float(mean_probs[order[1]]), 'top3_label': classes[order[2]], 'top3_prob': float(mean_probs[order[2]]), 'num_nodes': int(x.shape[0])}

            for class_name, probability in zip(classes, mean_probs):
                safe_name = safe_class_name(class_name)

                row[f'prob_{safe_name}'] = float(probability)

            for fold_index, fold in enumerate(args.folds):
                probs = fold_probabilities[fold_index]

                fold_pred_idx = int(np.argmax(probs))

                row[f'fold_{fold}_pred_label'] = classes[fold_pred_idx]

                row[f'fold_{fold}_pred_prob'] = float(probs[fold_pred_idx])
            rows.append(row)

            del (x, edge_index, edge_attr, fold_probabilities, mean_probs)
        except Exception as exc:
            failures.append({'graph': str(graph_path), 'error': str(exc), 'traceback': traceback.format_exc()})

    with open(output_dir / 'cptac_failures.json', 'w') as f:
        json.dump(failures, f, indent=2)
    slide_df = pd.DataFrame(rows)

    if slide_df.empty:
        raise RuntimeError('No CPTAC slides were evaluated successfully.')
    slide_df.to_csv(output_dir / 'cptac_slide_predictions.csv', index=False)

    slide_metrics = calculate_metrics(slide_df)

    project_rows = []

    for project, group in slide_df.groupby('project'):
        metrics = calculate_metrics(group)

        project_rows.append({'project': project, 'true_label': group['true_label'].iloc[0], 'n_slides': int(len(group)), 'top1': metrics['top1'], 'top3': metrics['top3'], 'weighted_f1': metrics['weighted_f1']})
    project_df = pd.DataFrame(project_rows).sort_values('project')

    project_df.to_csv(output_dir / 'cptac_per_project_metrics.csv', index=False)

    probability_columns = [f'prob_{safe_class_name(class_name)}' for class_name in classes]

    patient_rows = []

    for (project, patient_id), group in slide_df.groupby(['project', 'patient_id']):
        patient_probs = np.array([group[column].mean() for column in probability_columns], dtype=np.float64)

        order = np.argsort(patient_probs)[::-1]

        true_label = group['true_label'].iloc[0]

        true_idx = class_to_idx[true_label]

        patient_rows.append({'project': project, 'patient_id': patient_id, 'n_slides': int(len(group)), 'true_label': true_label, 'pred_label': classes[order[0]], 'pred_prob': float(patient_probs[order[0]]), 'correct': int(order[0] == true_idx), 'top3_correct': int(true_idx in order[:3]), 'top1_label': classes[order[0]], 'top2_label': classes[order[1]], 'top3_label': classes[order[2]]})
    patient_df = pd.DataFrame(patient_rows)

    patient_df.to_csv(output_dir / 'cptac_patient_predictions.csv', index=False)

    patient_metrics = calculate_metrics(patient_df)

    labels = sorted(set(slide_df['true_label']) | set(slide_df['pred_label']))

    cm = confusion_matrix(slide_df['true_label'], slide_df['pred_label'], labels=labels)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / 'cptac_confusion_matrix.csv')

    report = classification_report(slide_df['true_label'], slide_df['pred_label'], output_dict=True, zero_division=0)

    with open(output_dir / 'cptac_classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    summary = {'external_dataset': 'CPTAC', 'model': f'arithmetic-mean combination of {len(models)} TCGA fold Spatial KA-GAT models', 'folds': [int(fold) for fold in args.folds], 'fold_checkpoints': fold_metadata, 'ensemble_rule': 'For each slide, softmax probability vectors from all fold models are averaged element-wise. Patient-level probabilities are then obtained by averaging the ensemble slide probabilities across all slides for each patient.', 'cptac_used_for_training': False, 'cptac_used_for_model_selection': False, 'graphs_requested': int(len(graph_files)), 'slides_evaluated': int(len(slide_df)), 'failures': int(len(failures)), 'slide_level': slide_metrics, 'patient_level': patient_metrics, 'label_mapping': resolved_labels}

    with open(output_dir / 'cptac_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Slides evaluated: {len(slide_df)}/{len(graph_files)} | failures={len(failures)}')

    print(f"SLIDE:   Top-1={100 * slide_metrics['top1']:.2f}% Top-3={100 * slide_metrics['top3']:.2f}% Weighted F1={100 * slide_metrics['weighted_f1']:.2f}%")

    print(f"PATIENT: Top-1={100 * patient_metrics['top1']:.2f}% Top-3={100 * patient_metrics['top3']:.2f}% Weighted F1={100 * patient_metrics['weighted_f1']:.2f}%")

    print(project_df.to_string(index=False))

    print(f'OUTPUT ROOT: {output_dir}')


if __name__ == '__main__':
    main()
