# NEXUS

NEXUS is a spatial graph learning pipeline for cancer tissue-of-origin prediction using pathology foundation-model embeddings and spatial tissue organization.

The repository includes scripts for building spatial graphs, creating cross-validation splits, training spatial-aware and spatial-null models, generating spatial attention visualizations, and evaluating trained models on CPTAC using five-fold probability averaging.

## Repository Structure

NEXUS/
├── data/
│   ├── cptac/
│   │   ├── figures/
│   │   │   └── cptac_samples_per_tissue_origin.png
│   │   └── scripts/
│   │       └── build_cptac_graphs.py
│   │
│   └── tcga/
│       ├── figures/
│       │   └── tcga_samples_per_tissue_origin.png
│       ├── scripts/
│       ├── spatial-aware/
│       │   └── build_spatial_aware_graphs.py
│       ├── spatial-null/
│       │   └── build_spatial_null_graphs.py
│       ├── build_tcga_labels.py
│       └── create_cv_splits.py
│
├── interpretability/
│   ├── figures/
│   │   ├── spatial_aware_heatmap.png
│   │   └── spatial_null_heatmap.png
│   └── scripts/
│       └── plot_spatial_attention.py
│
├── models/
│   ├── spatial-aware/
│   │   ├── figures/
│   │   │   ├── confusion_matrix.png
│   │   │   ├── spatial-aware_vs_spatial-null.png
│   │   │   ├── train_accuracy_curve.png
│   │   │   ├── train_loss_curve.png
│   │   │   ├── validation_accuracy_curve.png
│   │   │   └── validation_loss_curve.png
│   │   └── scripts/
│   │       └── train_spatial_aware.py
│   │
│   └── spatial-null/
│       ├── figures/
│       │   ├── spatial-aware_vs_spatial-null.png
│       │   ├── train_accuracy_curve.png
│       │   ├── train_loss_curve.png
│       │   ├── validation_accuracy_curve.png
│       │   └── validation_loss_curve.png
│       └── scripts/
│           └── train_spatial_null.py
│
├── validation/
│   └── validate_cptac.py
│
└── LICENSE
```

## Pipeline

### 1. Build TCGA Spatial-Aware Graphs

`build_spatial_aware_graphs.py`

Builds spatial graphs from TCGA pathology embeddings and tile coordinates.

These graphs are used by:

- `build_tcga_labels.py`
- `create_cv_splits.py`
- `build_spatial_null_graphs.py`
- `train_spatial_aware.py`

### 2. Build TCGA Labels

`build_tcga_labels.py`

Creates the tumor-origin label metadata used for model training and evaluation.

### 3. Create Cross-Validation Splits

`create_cv_splits.py`

Creates the five cross-validation folds used by the spatial-aware and spatial-null training pipelines.

### 4. Build Spatial-Null Graphs

`build_spatial_null_graphs.py`

Creates the spatial-null control by permuting tile positions within each slide and rebuilding the spatial graph while preserving the underlying tile embeddings.

### 5. Train the Spatial-Aware Model

`train_spatial_aware.py`

Trains the spatial-aware NEXUS model using five-fold cross-validation.

The script also generates:

- Training loss curves
- Training accuracy curves
- Validation loss curves
- Validation accuracy curves
- Combined five-fold confusion matrix

### 6. Train the Spatial-Null Model

`train_spatial_null.py`

Trains the spatial-null control using the same five-fold framework.

The script also generates:

- Training loss curves
- Training accuracy curves
- Validation loss curves
- Validation accuracy curves
- Combined five-fold confusion matrix

### 7. Build CPTAC Graphs

`build_cptac_graphs.py`

Builds spatial graphs for CPTAC samples for external validation.

### 8. CPTAC External Validation

`validate_cptac.py`

Loads the five trained spatial-aware fold models and evaluates CPTAC samples.

For each CPTAC slide:

1. The slide is evaluated independently by all five fold models.
2. Each model produces a softmax probability vector.
3. The five probability vectors are averaged element-wise.
4. The class with the highest averaged probability is used as the final prediction.

For patients with multiple slides, the slide-level probability vectors are averaged again to produce a patient-level prediction.

Outputs include:

- Slide-level predictions
- Patient-level predictions
- Per-project metrics
- Confusion matrix
- Classification report
- Summary metrics

### 9. Spatial Attention Visualization

`plot_spatial_attention.py`

Generates spatial attention visualizations for a supplied graph, model checkpoint, and whole-slide image.

The script is patient-agnostic and can be used with any compatible sample.

## Main Graph Inputs

```text
Spatial-aware TCGA graphs
/workspace/data/tcga_spatial_graphs

Spatial-null TCGA graphs
/workspace/data/spatial_null_graphs

CPTAC graphs
/workspace/data/cptac_spatial_graphs
```

## Main Model Outputs

```text
Spatial-aware results
/workspace/results/NEXUS

Spatial-null results
/workspace/results/spatial_null

CPTAC validation results
/workspace/results/cptac

Spatial attention outputs
/workspace/results/spatial_attention
```

## Requirements

The code uses Python and commonly relies on:

```text
numpy
pandas
torch
scikit-learn
scipy
matplotlib
h5py
tqdm
requests
openslide-python
```

Install dependencies as needed for your environment.

## Notes

- The training pipeline uses five-fold cross-validation.
- CPTAC validation preserves the five-fold probability-averaging logic.
- Spatial-null experiments are intended to test the contribution of native spatial tissue organization.
- The scripts use descriptive `/workspace` paths and can also be redirected through command-line arguments where supported.

## License


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
