"""Evaluate the trained EDL model and visualize Bayesian uncertainty.

Runs the checkpoint over the FashionMNIST test split, reports accuracy and
mean Dirichlet uncertainty, and saves uncertainty_results.png: a side-by-side
figure where each test image is shown next to its Dirichlet predictive
distribution — per-class mean probability E[p_k] = alpha_k / S with a +/-1
sigma spread Var(p_k) = alpha_k (S - alpha_k) / (S^2 (S + 1)). Examples are
sampled across the test-set uncertainty percentiles, so the figure shows the
Dirichlet sharpening as evidence grows and flattening toward the uniform
prior (u -> 1) when the model has no evidence.

Usage:  python3 visualize.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EDLConvNet, dirichlet_uncertainty

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
# Uncertainty percentiles of the test set to showcase, low -> high.
PERCENTILES = [1, 25, 50, 75, 90, 97, 99.5, 100]


def get_device():
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError("MPS backend unavailable — this script must run on Apple Silicon MPS.")
    return torch.device("mps")


@torch.no_grad()
def infer_test_set(model, loader, device):
    """Collect Dirichlet alphas, predictions and uncertainties for all test images."""
    model.eval()
    all_alpha, all_labels, all_images = [], [], []
    for images, labels in loader:
        alpha = model(images.to(device))
        all_alpha.append(alpha.cpu())
        all_labels.append(labels)
        all_images.append(images)
    return torch.cat(all_alpha), torch.cat(all_labels), torch.cat(all_images)


def main():
    device = get_device()
    print(f"Device: {device} (Apple Silicon M2)")

    checkpoint = torch.load("edl_model.pth", map_location="cpu", weights_only=True)
    model = EDLConvNet(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)

    alpha, labels, images = infer_test_set(model, test_loader, device)
    S = alpha.sum(dim=1)
    probs, u = dirichlet_uncertainty(alpha)
    probs, u = probs.squeeze(1) if probs.dim() > 2 else probs, u.squeeze(1)
    preds = probs.argmax(dim=1)
    correct = preds == labels

    acc = correct.float().mean().item()
    print(f"Test accuracy:            {acc:.4f}")
    print(f"Mean uncertainty (all):   {u.mean().item():.4f}")
    print(f"Mean uncertainty (correct):   {u[correct].mean().item():.4f}")
    print(f"Mean uncertainty (mistakes):  {u[~correct].mean().item():.4f}")

    # Examples spanning the test-set uncertainty spectrum.
    order = torch.argsort(u)
    picks = []
    for pct in PERCENTILES:
        pos = min(len(order) - 1, max(0, round(pct / 100 * (len(order) - 1))))
        picks.append((int(order[pos]), f"P{pct:g}"))

    n_rows = len(picks)
    fig = plt.figure(figsize=(9.5, 2.35 * n_rows))
    gs = GridSpec(n_rows, 2, width_ratios=[1, 2.6], hspace=0.55, wspace=0.08, top=0.93)
    short_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

    for row, (idx, tag) in enumerate(picks):
        a = alpha[idx]
        s = S[idx].item()
        mean = (a / s).numpy()
        spread = torch.sqrt(a * (s - a) / (s**2 * (s + 1))).numpy()
        pred, true = preds[idx].item(), labels[idx].item()

        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(images[idx].squeeze(0), cmap="gray")
        verdict = "OK" if pred == true else "X"
        ax_img.set_title(
            f"[{tag}] Pred: {CLASSES[pred]} ({verdict})\n"
            f"True: {CLASSES[true]} | u = {u[idx]:.3f}",
            fontsize=9,
        )
        ax_img.axis("off")

        ax_bar = fig.add_subplot(gs[row, 1])
        colors = ["tab:gray"] * 10
        colors[pred] = "tab:green" if pred == true else "tab:red"
        ax_bar.bar(short_names, mean, yerr=spread, capsize=3, color=colors,
                   edgecolor="black", error_kw={"lw": 1.0})
        ax_bar.scatter([true], [1.02], marker="*", s=90, color="black", zorder=3)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel("Dirichlet\nE[p] +/- 1 sigma", fontsize=8)
        ax_bar.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax_bar.tick_params(axis="y", labelsize=8)
        ax_bar.set_title(
            f"Dirichlet strength S = {s:.1f}  ->  uncertainty u = K/S = 10/{s:.1f} = {u[idx]:.3f}",
            fontsize=9,
        )

    fig.suptitle(
        "Evidential Deep Learning on FashionMNIST — predicted class with Bayesian uncertainty\n"
        "Rows span the test-set uncertainty spectrum (P1 = most confident, P100 = most uncertain). "
        "Bars: Dirichlet predictive mean per class; whiskers: probability spread; star: true class.",
        fontsize=11,
    )
    fig.savefig("uncertainty_results.png", dpi=150, bbox_inches="tight")
    print("Saved figure -> uncertainty_results.png")


if __name__ == "__main__":
    main()
