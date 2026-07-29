#!/usr/bin/env python3
"""
Compare two MiniGPT-3D PointTransformer checkpoints on cls / patch features.

Zero-arg run (from MiniGPT-3D repo root):
  python point_model_VS_hybrid.py

Custom paths:
  python point_model_VS_hybrid.py \
    --ckpt-a ./params_weight/pc_encoder/point_model.pth \
    --ckpt-b ./params_weight/pc_encoder/point_model_pcpmae.pth

python point_model_VS_hybrid.py \
  --ckpt-a ./params_weight/pc_encoder/point_model.pth \
  --ckpt-b ./params_weight/pc_encoder/point_model_hybrid.pth \
  --data-path ./data/modelnet40_data/modelnet40_test_8192pts_fps.dat \
  --max-samples 2468
"""
import os
import sys
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from minigpt4.common.utils import cfg_from_yaml_file
from minigpt4.models.pointbert.point_encoder import PointTransformer


def resolve_default_ckpt(name, candidates):
    for rel in candidates:
        path = PROJECT_ROOT / rel
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"Cannot find default checkpoint for {name}. Tried:\n"
        + "\n".join(f"  - {PROJECT_ROOT / rel}" for rel in candidates)
    )


def resolve_default_data_path():
    candidates = [
        "data/modelnet40_data/modelnet40_test_8192pts_fps.dat",
        "data/modelnet40_data/modelnet40_test_8192pts_fps.pkl",
    ]
    for rel in candidates:
        path = PROJECT_ROOT / rel
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "Cannot find ModelNet40 test file. Expected one of:\n"
        + "\n".join(f"  - {PROJECT_ROOT / rel}" for rel in candidates)
    )


def pc_norm(pc):
    xyz, other = pc[:, :3], pc[:, 3:]
    xyz = xyz - xyz.mean(0)
    xyz = xyz / (np.sqrt((xyz ** 2).sum(1)).max() + 1e-8)
    return np.concatenate([xyz, other], axis=1).astype(np.float32)


def build_encoder(ckpt_path, device):
    yaml_path = PROJECT_ROOT / "minigpt4/models/pointbert/PointTransformer_8192point_2layer.yaml"
    cfg = cfg_from_yaml_file(str(yaml_path))
    cfg.model.point_dims = 6
    model = PointTransformer(cfg.model, use_max_pool=False).to(device)
    model.load_checkpoint(ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def extract(model, batch_pc, device):
    x = torch.from_numpy(batch_pc).to(device)
    feat = model(x)
    cls = feat[:, 0]
    patch = feat[:, 1:]
    global_patch = patch.max(dim=1).values
    router_feat = torch.cat([cls, global_patch], dim=-1)
    return {
        "cls": F.normalize(cls, dim=-1).cpu(),
        "global": F.normalize(global_patch, dim=-1).cpu(),
        "router": F.normalize(router_feat, dim=-1).cpu(),
        "patch_mean": F.normalize(patch.mean(dim=1), dim=-1).cpu(),
    }


def load_modelnet(dat_path, use_color=True, max_samples=None):
    with open(dat_path, "rb") as f:
        data = pickle.load(f)

  # PointLLM / MiniGPT-3D format: [point_clouds, labels]
    if isinstance(data, list) and len(data) == 2:
        point_clouds, labels = data
        pairs = list(zip(point_clouds, labels))
    elif isinstance(data, list):
        pairs = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append(item)
            else:
                raise ValueError(f"Unsupported list item format: {type(item)}")
    else:
        raise ValueError(f"Unsupported ModelNet dat format: {type(data)}")

    random.seed(42)
    if max_samples:
        pairs = random.sample(pairs, min(max_samples, len(pairs)))

    pcs, label_ids = [], []
    for pc, label in pairs:
        pc = np.array(pc, dtype=np.float32)
        if isinstance(label, np.ndarray):
            label = int(label.reshape(-1)[0])
        else:
            label = int(label)

        if pc.shape[1] == 3 and use_color:
            pc = np.concatenate(
                [pc, np.zeros((pc.shape[0], 3), dtype=np.float32)], axis=1
            )
        pcs.append(pc_norm(pc))
        label_ids.append(label)

    return np.stack(pcs), np.array(label_ids)


def pairwise_cos(a, b):
    return (a * b).sum(dim=-1).numpy()


def knn_eval(feat_np, labels, k=20, test_size=0.2):
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    min_class_count = int(class_counts[class_counts > 0].min()) if len(class_counts) else 0
    stratify = labels if min_class_count >= 2 else None
    if stratify is None:
        print("    [warn] too few samples per class for stratified split; using random split")

    X_tr, X_te, y_tr, y_te = train_test_split(
        feat_np,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    k = min(k, len(y_tr))
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))


def intra_inter_cos(feat_np, labels):
    from collections import defaultdict

    buckets = defaultdict(list)
    for feat, label in zip(feat_np, labels):
        buckets[label].append(feat)

    intra, inter = [], []
    classes = list(buckets.keys())
    rng = np.random.default_rng(42)

    for cls_id in classes:
        vecs = np.stack(buckets[cls_id])
        if len(vecs) < 2:
            continue
        idx = rng.choice(len(vecs), size=min(30, len(vecs)), replace=False)
        sub = vecs[idx]
        sim = sub @ sub.T
        intra.extend(sim[np.triu_indices(len(sub), k=1)])

    for _ in range(500):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        v1 = buckets[c1][rng.integers(len(buckets[c1]))]
        v2 = buckets[c2][rng.integers(len(buckets[c2]))]
        inter.append(float(v1 @ v2))

    return float(np.mean(intra)), float(np.mean(inter))


def compare_state_dict(ckpt_a, ckpt_b, topk=20):
    sa = torch.load(ckpt_a, map_location="cpu")["base_model"]
    sb = torch.load(ckpt_b, map_location="cpu")["base_model"]
    keys = sorted(set(sa.keys()) | set(sb.keys()))

    print("\n=== Parameter-level cosine (top differences) ===")
    rows = []
    for key in keys:
        if key not in sa:
            print(f"  missing in A: {key}")
            continue
        if key not in sb:
            print(f"  missing in B: {key}")
            continue
        if sa[key].shape != sb[key].shape:
            print(f"  shape mismatch: {key}  A={tuple(sa[key].shape)}  B={tuple(sb[key].shape)}")
            continue
        cos = F.cosine_similarity(
            sa[key].flatten().float(), sb[key].flatten().float(), dim=0
        ).item()
        rows.append((key, cos))

    rows.sort(key=lambda x: x[1])
    for key, cos in rows[:topk]:
        print(f"  {key:50s} cos={cos:.4f}")
    if rows:
        mean_cos = float(np.mean([cos for _, cos in rows]))
        print(f"  {'[mean over shared keys]':50s} cos={mean_cos:.4f}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare two PointTransformer checkpoints for MiniGPT-3D."
    )
    parser.add_argument(
        "--ckpt-a",
        default=None,
        help="Baseline checkpoint (default: params_weight/pc_encoder/point_model.pth)",
    )
    parser.add_argument(
        "--ckpt-b",
        default=None,
        help=(
            "Candidate checkpoint (default: first existing among "
            "point_model_hybrid.pth / point_model_pcpmae.pth / pcpmae_ShapeNet.pth)"
        ),
    )
    parser.add_argument(
        "--data-type",
        choices=["modelnet", "objaverse"],
        default="modelnet",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="ModelNet .dat/.pkl path (default: data/modelnet40_data/modelnet40_test_8192pts_fps.dat)",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--compare-weights",
        action="store_true",
        help="Also print per-parameter cosine between A and B",
    )
    return parser


def resolve_args(args):
    args.ckpt_a = args.ckpt_a or resolve_default_ckpt(
        "ckpt-a",
        ["params_weight/pc_encoder/point_model.pth"],
    )
    args.ckpt_b = args.ckpt_b or resolve_default_ckpt(
        "ckpt-b",
        [
            "params_weight/pc_encoder/point_model_hybrid.pth",
            "params_weight/pc_encoder/point_model_pcpmae.pth",
            "params_weight/pc_encoder/pcpmae_ShapeNet.pth",
        ],
    )
    if args.data_type == "modelnet":
        args.data_path = args.data_path or resolve_default_data_path()
    elif args.data_path is None:
        raise ValueError("Objaverse mode requires --data-path")
    return args


def main():
    args = resolve_args(build_arg_parser().parse_args())

    print("=== Config ===")
    print(f"  ckpt A (baseline):  {args.ckpt_a}")
    print(f"  ckpt B (candidate): {args.ckpt_b}")
    print(f"  data:               {args.data_path}")
    print(f"  max_samples:        {args.max_samples}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device:             {device}")

    enc_a = build_encoder(args.ckpt_a, device)
    enc_b = build_encoder(args.ckpt_b, device)

    if args.data_type == "modelnet":
        pcs, labels = load_modelnet(args.data_path, max_samples=args.max_samples)
    else:
        raise NotImplementedError("Objaverse mode is not implemented yet.")

    all_a = {k: [] for k in ["cls", "global", "router", "patch_mean"]}
    all_b = {k: [] for k in ["cls", "global", "router", "patch_mean"]}

    for i in range(0, len(pcs), args.batch_size):
        batch = pcs[i : i + args.batch_size]
        fa = extract(enc_a, batch, device)
        fb = extract(enc_b, batch, device)
        for key in all_a:
            all_a[key].append(fa[key])
            all_b[key].append(fb[key])

    for key in all_a:
        all_a[key] = torch.cat(all_a[key], dim=0)
        all_b[key] = torch.cat(all_b[key], dim=0)

    print("\n=== Same-sample cosine (A vs B) ===")
    for name in ["cls", "global", "router", "patch_mean"]:
        cos = pairwise_cos(all_a[name], all_b[name])
        print(
            f"  {name:12s}: mean={cos.mean():.4f}, min={cos.min():.4f}, "
            f"p10={np.percentile(cos, 10):.4f}"
        )

    print("\n=== kNN accuracy on ModelNet40 (stratified split) ===")
    labels = labels[: len(all_a["cls"])]
    for name in ["cls", "global", "router"]:
        acc_a = knn_eval(all_a[name].numpy(), labels)
        acc_b = knn_eval(all_b[name].numpy(), labels)
        print(
            f"  {name:12s}: A={acc_a * 100:.2f}%  B={acc_b * 100:.2f}%  "
            f"(gap={acc_b - acc_a:+.2%})"
        )

    print("\n=== Intra-class vs Inter-class cosine ===")
    for tag, feats in [("A", all_a), ("B", all_b)]:
        intra, inter = intra_inter_cos(feats["cls"].numpy(), labels)
        print(
            f"  Encoder {tag} cls: intra={intra:.4f}, inter={inter:.4f}, "
            f"margin={intra - inter:.4f}"
        )

    if args.compare_weights:
        compare_state_dict(args.ckpt_a, args.ckpt_b)


if __name__ == "__main__":
    main()
