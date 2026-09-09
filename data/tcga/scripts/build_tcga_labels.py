#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_TO_TUMOR_ORIGIN = {'TCGA-BRCA': 'Breast', 'TCGA-LUAD': 'Lung', 'TCGA-LUSC': 'Lung', 'TCGA-GBM': 'Brain', 'TCGA-LGG': 'Brain', 'TCGA-KICH': 'Kidney', 'TCGA-KIRC': 'Kidney', 'TCGA-KIRP': 'Kidney', 'TCGA-UCEC': 'Uterus', 'TCGA-UCS': 'Uterus', 'TCGA-HNSC': 'Head and Neck', 'TCGA-THCA': 'Thyroid', 'TCGA-COAD': 'Colon', 'TCGA-PRAD': 'Prostate', 'TCGA-BLCA': 'Bladder', 'TCGA-STAD': 'Stomach', 'TCGA-LIHC': 'Liver', 'TCGA-SKCM': 'Skin', 'TCGA-CESC': 'Cervix', 'TCGA-SARC': 'Soft Tissue', 'TCGA-ACC': 'Adrenal Gland/Paraganglia', 'TCGA-PCPG': 'Adrenal Gland/Paraganglia', 'TCGA-PAAD': 'Pancreas', 'TCGA-ESCA': 'Esophagus', 'TCGA-TGCT': 'Testis', 'TCGA-READ': 'Rectum', 'TCGA-THYM': 'Thymus', 'TCGA-UVM': 'Eye', 'TCGA-MESO': 'Pleura/Mesothelium', 'TCGA-OV': 'Ovary', 'TCGA-DLBC': 'Lymphatic System', 'TCGA-CHOL': 'Bile Duct'}


def normalize_patient_id(name):
    stem = Path(str(name)).stem
    parts = stem.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3])

    return stem


def extract_tss(patient_id):
    parts = patient_id.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return parts[1]

    return 'UNKNOWN'


def discover_graph_patients(graph_dir):
    graph_dir = Path(graph_dir)

    paths = sorted(graph_dir.glob('*.pt'))

    if not paths:
        raise RuntimeError(f'No .pt graph files found in {graph_dir}')
    patients = sorted({normalize_patient_id(path.name) for path in paths})

    return patients


def query_gdc_chunk(patient_ids, timeout=120):
    url = 'https://api.gdc.cancer.gov/cases'
    filters = {'op': 'in', 'content': {'field': 'submitter_id', 'value': patient_ids}}

    payload = {'filters': json.dumps(filters), 'fields': 'submitter_id,project.project_id', 'format': 'JSON', 'size': str(max(len(patient_ids), 1))}

    response = requests.post(url, data=payload, timeout=timeout)

    response.raise_for_status()

    data = response.json()

    hits = data.get('data', {}).get('hits', [])

    rows = []

    for hit in hits:
        submitter_id = hit.get('submitter_id')

        project = hit.get('project', {})

        project_id = project.get('project_id')

        rows.append({'patient_norm': normalize_patient_id(submitter_id), 'project_id': project_id})

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph-dir', default='/workspace/data/tcga_spatial_graphs')

    parser.add_argument('--output', default='/workspace/metadata/tcga_labels.csv')

    parser.add_argument('--chunk-size', type=int, default=400)

    parser.add_argument('--retries', type=int, default=5)

    args = parser.parse_args()

    patients = discover_graph_patients(args.graph_dir)

    print(f'Graph patients: {len(patients):,}')

    rows = []

    chunks = [patients[i:i + args.chunk_size] for i in range(0, len(patients), args.chunk_size)]

    for chunk in tqdm(chunks, desc='Querying GDC', unit='chunk'):
        last_error = None

        for attempt in range(1, args.retries + 1):
            try:
                rows.extend(query_gdc_chunk(chunk))

                last_error = None
                break
            except Exception as exc:
                last_error = exc

                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 15))

        if last_error is not None:
            raise RuntimeError(f'GDC query failed after {args.retries} attempts: {last_error}')
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError('GDC returned zero cases.')
    df = df.drop_duplicates(subset=['patient_norm'], keep='first').copy()

    df['label_text'] = df['project_id'].map(PROJECT_TO_TUMOR_ORIGIN)

    df['site'] = df['patient_norm'].map(extract_tss)

    graph_patient_set = set(patients)

    returned_patient_set = set(df['patient_norm'].astype(str))

    missing_from_gdc = sorted(graph_patient_set - returned_patient_set)

    unknown_project = df[df['label_text'].isna()].copy()

    if missing_from_gdc:
        print('ERROR: graph patients missing from GDC:', len(missing_from_gdc))

        print('Examples:', missing_from_gdc[:20])

    if not unknown_project.empty:
        print('ERROR: unsupported TCGA project IDs:', sorted(unknown_project['project_id'].dropna().unique().tolist()))

    if missing_from_gdc or not unknown_project.empty:
        diagnostics_dir = Path(args.output).parent
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        if missing_from_gdc:
            pd.DataFrame({'patient_norm': missing_from_gdc}).to_csv(diagnostics_dir / 'missing_from_gdc.csv', index=False)

        if not unknown_project.empty:
            unknown_project.to_csv(diagnostics_dir / 'unknown_projects.csv', index=False)

        raise RuntimeError('Could not safely label every graph patient. See diagnostic CSV files.')
    df = df[['patient_norm', 'label_text', 'project_id', 'site']].sort_values('patient_norm')

    if len(df) != len(patients):
        raise RuntimeError(f'Label count mismatch: {len(df):,} labels vs {len(patients):,} graph patients.')
    output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)

    tumor_counts = df['label_text'].value_counts().sort_values(ascending=False)

    project_counts = df['project_id'].value_counts().sort_values(ascending=False)

    site_counts = df['site'].value_counts().sort_values(ascending=False)

    tumor_counts.to_csv(output.parent / 'tumor_origin_counts.csv', header=['count'])

    project_counts.to_csv(output.parent / 'tcga_project_counts.csv', header=['count'])

    site_counts.to_csv(output.parent / 'tcga_site_counts.csv', header=['count'])

    print(f'Total labeled patients: {len(df):,}')

    print(f"Tumor origins:          {df['label_text'].nunique()}")

    print(f"TCGA projects:          {df['project_id'].nunique()}")

    print(f"TCGA TSS sites:         {df['site'].nunique()}")

    print(tumor_counts.to_string())

    print(project_counts.to_string())

    print(f'Saved labels: {output}')

    print('Saved tumor counts:', output.parent / 'tumor_origin_counts.csv')

    print('Saved project counts:', output.parent / 'tcga_project_counts.csv')

    print('Saved site counts:', output.parent / 'tcga_site_counts.csv')


if __name__ == '__main__':
    main()
