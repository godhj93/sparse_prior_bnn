from __future__ import annotations

import os  
import argparse
import random
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import spearmanr
import umap.umap_ as umap
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# from utils.models.densenet_uni import densenet_bc_30_uni
# from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_uni
from torchvision.datasets import ImageFolder

# try:
#     from bayesian_torch.layers.variational_layers.conv_variational import (
#         Conv2dReparameterization,
#         Conv2dReparameterization_Multivariate,
#     )
# except ModuleNotFoundError:  # noqa: F401 – placeholder to keep import order
#     Conv2dReparameterization = Conv2dReparameterization_Multivariate = nn.Module


IMG_STATS = {
    "TinyImageNet": ((0.4802, 0.4481, 0.3975),
                     (0.2302, 0.2265, 0.2262)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
}


CLASSIFICATION_DATASETS = {
    "CIFAR10": lambda root: datasets.CIFAR10(
        root=root,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*IMG_STATS["CIFAR10"]),
        ]),
        download=True,
    ),

    # ★ 여기 root 경로 수정:  .../val  (images 아님!)
    "TinyImageNet": lambda root: ImageFolder(
        root=os.path.join(root, "tiny-imagenet-200", "val"),
        transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(*IMG_STATS["TinyImageNet"]),
        ]),
    ),
}

@torch.no_grad()
def extract_features_resnet(
    model: nn.Module,
    loader: DataLoader,
    mc_runs: int,
    device: str = "cpu",
):
    """Return features [N,D] and labels [N] with tqdm progress."""
    model.eval()
    feats: List[torch.Tensor] = []
    lbls: List[int] = []

    def _hook(_, inp, __):  # inp[0] shape: [B, D]
        feats.append(inp[0].detach().cpu())

    # Register hook on final linear layer (classifier)
    h = model.linear.register_forward_hook(_hook)

    total_iter = len(loader) * mc_runs
    with tqdm(total=total_iter, desc="Extracting", unit="sample") as pbar:
        for images, targets in loader:
            images = images.to(device)
            for _ in range(mc_runs):
                model(images)
                lbls.extend(targets.numpy().tolist())
                pbar.update(1)

    h.remove()
    return torch.cat(feats, 0), torch.tensor(lbls)

@torch.no_grad()
def extract_features_densenet(
    model: nn.Module,
    loader: DataLoader,
    mc_runs: int,
    device: str = "cpu",
):
    """Return features [N,D] and labels [N] with tqdm progress."""
    model.eval()
    feats: List[torch.Tensor] = []
    lbls: List[int] = []

    def _hook(_, inp, __):  # inp[0] shape: [B, D]
        feats.append(inp[0].detach().cpu())

    h = model.classifier.register_forward_hook(_hook)

    total_iter = len(loader) * mc_runs
    with tqdm(total=total_iter, desc="Extracting", unit="sample") as pbar:
        for images, targets in loader:
            images = images.to(device)
            for _ in range(mc_runs):
                model(images)
                lbls.extend(targets.numpy().tolist())
                pbar.update(1)

    h.remove()
    return torch.cat(feats, 0), torch.tensor(lbls)

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    mc_runs: int,
    device: str = "cpu",
):
    """Return features [N,D] and labels [N] with tqdm progress."""
    model.eval()
    feats: List[torch.Tensor] = []
    lbls: List[int] = []

    def _hook(_, inp, __):  # inp[0] shape: [B, D]
        feats.append(inp[0].detach().cpu())

    # h = model[-1].register_forward_hook(_hook)
    try:
        classifier = model.base_model.head  # Should match the input features of the head
    except:
        classifier = list(model.children())[-1]  # last layer
        
    h = classifier.register_forward_hook(_hook)

    total_iter = len(loader) * mc_runs
    with tqdm(total=total_iter, desc="Extracting", unit="sample") as pbar:
        for images, targets in loader:
            images = images.to(device)
            for _ in range(mc_runs):
                model(images)
                lbls.extend(targets.numpy().tolist())
                pbar.update(1)

    h.remove()
    return torch.cat(feats, 0), torch.tensor(lbls)


def _nearest_rank_matrix(X: np.ndarray, k_max: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k_max + 1, n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    N = X.shape[0]
    rank = np.full((N, N), N, int)
    for i, nbrs in enumerate(indices):
        for r, j in enumerate(nbrs[1:], 1):  # skip self
            rank[i, j] = r
    return rank


def continuity(high_rank: np.ndarray, low_rank: np.ndarray, k: int) -> float:
    n = high_rank.shape[0]
    penalty = 0.0
    for i in range(n):
        high_neigh = np.where(high_rank[i] <= k)[0]
        low_rank_i = low_rank[i]
        penalty += np.maximum(0, low_rank_i[high_neigh] - k).sum()
    return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * penalty


def q_nx(T: float, C: float, k: int, n: int) -> float:
    return (T + C) / 2 - k / (n - 1)


# ---- embedding runner ---- #

def run_embedding(X: np.ndarray, *, method: str, **kwargs):
    if method == "tsne":
        tsne = TSNE(n_components=2, random_state=42, perplexity=kwargs.get("perplexity", 30))
        Y = tsne.fit_transform(X)
        return Y, tsne.kl_divergence_
    if method == "umap":
        um = umap.UMAP(
            n_components=2,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric="euclidean",
            random_state=42,
        )
        return um.fit_transform(X), None
    raise ValueError("method must be 'tsne' or 'umap'")

# ---- cluster / global metrics ---- #

def participation_ratio(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-8]  # remove near-zero
    return (eig.sum() ** 2) / (eig ** 2).sum()



def generalised_variance(cov: np.ndarray) -> float:
    jitter = 1e-8 * np.eye(cov.shape[0])
    return float(np.linalg.det(cov + jitter))


def cluster_metrics(Y: np.ndarray, labels: np.ndarray):
    sil = silhouette_score(Y, labels)
    db = davies_bouldin_score(Y, labels)

    pr_list, gv_list = [], []
    for cls in np.unique(labels):
        pts = Y[labels == cls]
        if pts.shape[0] < 2:
            continue
        cov = np.cov(pts.T)
        pr_list.append(participation_ratio(cov))
        gv_list.append(generalised_variance(cov))
    pr = float(np.mean(pr_list)) if pr_list else np.nan
    gv = float(np.mean(gv_list)) if gv_list else np.nan
    return sil, db, pr, gv


def global_rank_corr(X: np.ndarray, Y: np.ndarray, sample: int = 2000):
    n = X.shape[0]
    idx = np.random.choice(n, min(sample, n), replace=False)
    dX = np.linalg.norm(X[idx][:, None] - X[idx][None], axis=-1).ravel()
    dY = np.linalg.norm(Y[idx][:, None] - Y[idx][None], axis=-1).ravel()
    rho, _ = spearmanr(dX, dY)
    return rho

# -------------------------------------------------------------------------------- #
# Plotting
# -------------------------------------------------------------------------------- #

def scatter_plot(Y: np.ndarray, labels: torch.Tensor, path: Path):
    plt.figure(figsize=(8, 8))
    palette = sns.color_palette("tab10", n_colors=len(labels.unique()))
    for cls in range(len(palette)):
        idx = (labels == cls).numpy()
        plt.scatter(Y[idx, 0], Y[idx, 1], s=6, color=palette[cls], alpha=0.6, rasterized=True, label=str(cls))
    plt.axis("off")
    plt.tight_layout()
    plt.legend(title="Class", markerscale=2, fontsize="small")
    plt.savefig(path, dpi=300)
    plt.close()

# -------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------- #

def main():
    #* Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["densenet30", "resnet20"], default="resnet20")
    p.add_argument("--data", choices=CLASSIFICATION_DATASETS.keys(), required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--mc_runs", type=int, default=10)
    p.add_argument("--samples_per_class", type=int, default=50)
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--perplexity", type=int, default=30)          # t‑SNE
    p.add_argument("--n_neighbors", type=int, default=15)         # UMAP
    p.add_argument("--min_dist", type=float, default=0.1)         # UMAP
    p.add_argument("--folder", default="embed_out") # output folder for plots
    p.add_argument("--name", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CLASSIFICATION_DATASETS[args.data]("./data")
    num_classes = len(dataset.classes)

    counter = {i: 0 for i in range(num_classes)}
    sel_idx: List[int] = []
    for idx, (_, lbl) in tqdm(enumerate(dataset), total=len(dataset), desc="Sampling"):
        if counter[lbl] < args.samples_per_class:
            sel_idx.append(idx)
            counter[lbl] += 1
        if all(v >= args.samples_per_class for v in counter.values()):
            break

    loader = DataLoader(Subset(dataset, sel_idx), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    #* select model
    if args.model == "densenet30":

        model = densenet_bc_30_uni(num_classes = num_classes).to(device)
        model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)

        feats, lbls = extract_features_densenet(model, loader, args.mc_runs, device)
        #* [N, D] features and [N] labels

        Y, kl = run_embedding(
            feats.numpy(),
            method=args.method,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        sil, db, pr, gv = cluster_metrics(Y, lbls.numpy())
        rho = global_rank_corr(feats.numpy(), Y)

        print("\n=== Metrics ===")
        print(f"Silhouette={sil:.3f}  DaviesBouldin={db:.3f}  PR={pr:.2f}  GV={gv:.2e}  Spearmanρ={rho:.3f}\n")

        # out_dir = Path(args.folder)
        # out_dir.mkdir(parents=True, exist_ok=True)
        # prefix = f"{args.name}_" if args.name else ""
        # scatter_plot(Y, lbls, out_dir / f"{prefix}{args.method}.png")
        # print("[+] Saved plot →", (out_dir / f"{prefix}{args.method}.png").resolve())
    
    elif args.model == "resnet20":

        model = resnet20_uni(num_classes = num_classes).to(device)
        model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)

        feats, lbls = extract_features_resnet(model, loader, mc_runs=args.mc_runs, device="cuda")

        Y, kl = run_embedding(
            feats.numpy(),
            method=args.method,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        sil, db, pr, gv = cluster_metrics(Y, lbls.numpy())
        rho = global_rank_corr(feats.numpy(), Y)

        print("\n=== Metrics ===")
        print(f"Silhouette={sil:.3f}  DaviesBouldin={db:.3f}  PR={pr:.2f}  GV={gv:.2e}  Spearmanρ={rho:.3f}\n")

        # out_dir = Path(args.folder)
        # out_dir.mkdir(parents=True, exist_ok=True)
        # prefix = f"{args.name}_" if args.name else ""
        # scatter_plot(Y, lbls, out_dir / f"{prefix}{args.method}.png")
        # print("[+] Saved plot →", (out_dir / f"{prefix}{args.method}.png").resolve())

def run_tsne(model, dataset, device, args):
    
    model.eval()
    


if __name__ == "__main__":
    main()
