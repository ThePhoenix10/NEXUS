#!/usr/bin/env python3

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

CMAP = 'inferno'
RENDER_LEVEL = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph', required=True)

    parser.add_argument('--ckpt', required=True)

    parser.add_argument('--wsi', required=True)

    parser.add_argument('--train_script', required=True)

    parser.add_argument('--out_dir', default='/workspace/results/spatial_attention')

    parser.add_argument('--label', default=None)

    parser.add_argument('--dpi', type=int, default=600)

    parser.add_argument('--top_k', type=int, default=200)

    return parser.parse_args()


def import_model_class(train_script: str):
    spec = importlib.util.spec_from_file_location('train_module', train_script)

    mod = importlib.util.module_from_spec(spec)

    sys.modules['train_module'] = mod
    spec.loader.exec_module(mod)

    if hasattr(mod, 'SpatialGATKAN'):
        return getattr(mod, 'SpatialGATKAN')

    raise RuntimeError('SpatialGATKAN not found in train script')


def load_model(ModelClass, ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, nn.Module):
        return ckpt.to(device).eval()
    sd = ckpt['model_state_dict']

    cfg = dict(input_dim=ckpt['input_dim'], hidden_dim=ckpt['hidden_dim'], num_classes=len(ckpt['classes']), num_layers=ckpt['num_layers'], num_heads=ckpt['num_heads'], kan_grid_size=ckpt['kan_grid_size'], radius_multiplier=ckpt['radius_multiplier'], kan_rank=ckpt['kan_rank'], kan_frequencies=ckpt['kan_frequencies'], dropout=ckpt['dropout'])

    model = ModelClass(**cfg)

    model.load_state_dict(sd, strict=True)

    model = model.to(device).eval()

    model.class_names = list(ckpt.get('classes', []))

    return model


def extract_attention(model: nn.Module, graph: dict, device: torch.device) -> np.ndarray:
    x = graph['x'].float().to(device)

    edge_index = graph['edge_index'].long().to(device)

    edge_attr = graph['edge_attr'].float().to(device)

    n_nodes = x.shape[0]

    batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

    captured = {}

    pool = model.attention_pool
    original_forward = pool.forward


    def patched_forward(x_, batch_, num_graphs_):
        pooled, weights = original_forward(x_, batch_, num_graphs_)

        captured['weights'] = weights.detach()

        return (pooled, weights)
    pool.forward = patched_forward

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, batch)
    pool.forward = original_forward
    weights = captured['weights'].cpu().numpy().astype(np.float32)

    assert len(weights) == n_nodes

    return (weights, logits.detach().cpu())


def load_wsi_thumbnail(wsi_path: str):
    import openslide
    slide = openslide.OpenSlide(wsi_path)

    render_level = min(RENDER_LEVEL, slide.level_count - 1)

    level0_w, level0_h = slide.level_dimensions[0]

    render_w, render_h = slide.level_dimensions[render_level]

    region = slide.read_region((0, 0), render_level, (render_w, render_h))

    thumbnail = np.array(region.convert('RGB'))

    slide.close()

    return (thumbnail, level0_w, level0_h, render_w, render_h)


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    if len(arr) <= 1:
        return np.zeros_like(arr, dtype=np.float32)

    return (rankdata(arr) - 1) / (len(arr) - 1)


def _build_heatmap(x_rl, y_rl, attn_pct, canvas_w, canvas_h, tile_px):
    hmap = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)

    tile_size = max(1, int(round(tile_px)))

    xi = np.clip(np.round(x_rl).astype(int), 0, canvas_w - 1)

    yi = np.clip(np.round(y_rl).astype(int), 0, canvas_h - 1)

    order = np.argsort(attn_pct)

    for idx in order:
        row_start = yi[idx]

        row_end = min(row_start + tile_size, canvas_h)

        col_start = xi[idx]

        col_end = min(col_start + tile_size, canvas_w)

        hmap[row_start:row_end, col_start:col_end] = attn_pct[idx]
    mask = ~np.isnan(hmap)

    hmap = np.nan_to_num(hmap, nan=0.0)

    return (hmap, mask)


def make_figure(graph: dict, attn: np.ndarray, thumb: np.ndarray, patient_id: str, primary_label: str, render_w: int, render_h: int, level0_w: int, out_dir: Path, dpi: int, top_k: int):
    from scipy.ndimage import gaussian_filter
    pos = graph['pos'].numpy().astype(np.float32)

    x_l0 = pos[:, 0]

    y_l0 = pos[:, 1]

    level0_to_render = level0_w / render_w
    x_rl = x_l0 / level0_to_render
    y_rl = y_l0 / level0_to_render
    patch_size_level0 = float(graph.get('tile_stride', 512.0))

    tile_px = patch_size_level0 / level0_to_render
    edge_index = graph['edge_index'].numpy()

    attn_pct = _percentile_rank(attn)

    cmap = plt.get_cmap(CMAP)

    norm = Normalize(vmin=0, vmax=1)

    scalar_map = ScalarMappable(cmap=cmap, norm=norm)

    scalar_map.set_array([])

    canvas_h = int(render_h) + 1
    canvas_w = int(render_w) + 1
    hmap, tissue_mask = _build_heatmap(x_rl, y_rl, attn_pct, canvas_w, canvas_h, tile_px)

    hmap_smooth = gaussian_filter(hmap, sigma=tile_px * 3.0)

    tissue_values = hmap_smooth[tissue_mask]

    if len(tissue_values) > 0:
        vmin = tissue_values.min()

        vmax = tissue_values.max()

        hmap_smooth = (hmap_smooth - vmin) / (vmax - vmin + 1e-12)
    rgba = cmap(hmap_smooth)

    rgba[..., 3] = np.where(tissue_mask, 0.6, 0.0)

    src_e = edge_index[0]

    dst_e = edge_index[1]

    edge_score = attn_pct[src_e] * attn_pct[dst_e]

    edge_norm = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min() + 1e-12)

    threshold = np.percentile(edge_norm, 80)

    keep = edge_norm >= threshold
    src_k = src_e[keep]

    dst_k = dst_e[keep]

    score_k = edge_norm[keep]

    segs_wsi = np.stack([np.stack([x_rl[src_k], y_rl[src_k]], axis=1), np.stack([x_rl[dst_k], y_rl[dst_k]], axis=1)], axis=1)

    edge_cmap = plt.get_cmap('inferno')

    all_colors = edge_cmap(score_k).copy()

    all_colors[:, 3] = 0.75 + score_k * 0.25
    all_lw = np.full(len(score_k), 0.2, dtype=np.float32)

    fig, (ax_hm, ax_edge) = plt.subplots(1, 2, figsize=(22, 9), facecolor='white', gridspec_kw={'wspace': 0.14})

    suptitle = f'Patient: {patient_id}   |   Primary Origin: {primary_label}   |   Spatial - Aware'
    fig.suptitle(suptitle, fontsize=24, fontweight='bold', y=0.93)

    ax_hm.imshow(thumb, origin='upper', interpolation='lanczos', extent=[0, render_w, render_h, 0], zorder=0)

    ax_hm.imshow(rgba, origin='upper', extent=[0, canvas_w, canvas_h, 0], interpolation='bilinear', zorder=1)

    ax_hm.contour(tissue_mask.astype(float), levels=[0.5], colors=['#333333'], linewidths=[0.4], origin='upper', extent=[0, canvas_w, canvas_h, 0], zorder=2)

    ax_hm.set_xlim(0, render_w)

    ax_hm.set_ylim(render_h, 0)

    ax_hm.set_aspect('equal')

    ax_hm.axis('off')

    ax_hm.set_title('A  —  Node Attention Heatmap', fontsize=20, loc='center', pad=24, fontweight='bold')

    cb1 = fig.colorbar(scalar_map, ax=ax_hm, fraction=0.022, pad=0.01, shrink=0.72)

    cb1.set_label('Attention Score', fontsize=15, fontweight='bold', labelpad=12)

    cb1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    cb1.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    cb1.ax.tick_params(labelsize=13, width=1.2)

    for label in cb1.ax.get_yticklabels():
        label.set_fontweight('bold')
    thumb_blend = (thumb * 0.5).astype(np.uint8)

    ax_edge.imshow(thumb_blend, origin='upper', interpolation='lanczos', extent=[0, render_w, render_h, 0], zorder=0)

    line_collection = LineCollection(segs_wsi, linewidths=all_lw, colors=all_colors, rasterized=True, zorder=1, capstyle='round')

    ax_edge.add_collection(line_collection)

    ax_edge.set_xlim(0, render_w)

    ax_edge.set_ylim(render_h, 0)

    ax_edge.set_aspect('equal')

    ax_edge.axis('off')

    ax_edge.set_title('B  —  Edge Attention Heatmap', fontsize=20, loc='center', pad=24, fontweight='bold')

    scalar_map_edge = ScalarMappable(cmap=plt.get_cmap('inferno'), norm=norm)

    scalar_map_edge.set_array([])

    cb2 = fig.colorbar(scalar_map_edge, ax=ax_edge, fraction=0.022, pad=0.01, shrink=0.72)

    cb2.set_label('Edge Attention Score', fontsize=15, fontweight='bold', labelpad=12)

    cb2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    cb2.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    cb2.ax.tick_params(labelsize=13, width=1.2)

    for label in cb2.ax.get_yticklabels():
        label.set_fontweight('bold')
    fig.text(0.5, 0.16, f'$\\mathbf{{Nodes = {len(attn):,}}}$     $\\mathbf{{Edges = {edge_index.shape[1]:,}}}$', ha='center', va='bottom', fontsize=14, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75', lw=0.8))

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_dir / f'{patient_id}_spatial_attention'
    fig.savefig(str(stem) + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')

    fig.savefig(str(stem) + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')

    plt.close(fig)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = torch.load(args.graph, map_location='cpu', weights_only=False)

    if 'x' not in graph:
        import h5py
        h5_path = str(graph['source_h5'])

        feature_key = str(graph.get('feature_key', 'features'))

        node_indices = graph['node_indices'].numpy().astype(int)

        with h5py.File(h5_path, 'r') as h5_file:
            raw = h5_file[feature_key][:]

            all_features = raw[0] if raw.ndim == 3 else raw
            features = all_features[node_indices]
        graph['x'] = torch.tensor(features, dtype=torch.float32)
    ModelClass = import_model_class(args.train_script)

    model = load_model(ModelClass, args.ckpt, device)

    attn, logits = extract_attention(model, graph, device)

    probabilities = torch.softmax(logits, dim=1)[0].numpy()

    pred_index = int(np.argmax(probabilities))

    pred_label = model.class_names[pred_index] if model.class_names else str(pred_index)

    patient_id = str(graph.get('patient_id', Path(args.graph).stem))

    primary_label = args.label or str(graph.get('label_text', graph.get('label', pred_label)))

    thumb, level0_w, level0_h, render_w, render_h = load_wsi_thumbnail(args.wsi)

    make_figure(graph=graph, attn=attn, thumb=thumb, patient_id=patient_id, primary_label=primary_label, render_w=render_w, render_h=render_h, level0_w=level0_w, out_dir=Path(args.out_dir), dpi=args.dpi, top_k=args.top_k)


if __name__ == '__main__':
    main()
