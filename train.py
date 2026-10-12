"""Train the EDL (Dirichlet) classifier on FashionMNIST using the Apple MPS backend.

Automatically downloads FashionMNIST into ./data, trains for a few epochs,
saves the checkpoint to edl_model.pth, and plots the training loss and
accuracy curves to training_metrics.png.

Usage:  python3 train.py [--epochs 8] [--batch-size 128] [--lr 1e-3]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EDLConvNet, dirichlet_uncertainty, edl_digamma_loss, edl_mse_loss

NUM_CLASSES = 10


def get_device():
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError("MPS backend unavailable — this script must run on Apple Silicon MPS.")
    return torch.device("mps")


def get_dataloaders(batch_size):
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, uncertainty_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        alpha = model(images)
        probs, uncertainty = dirichlet_uncertainty(alpha)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        uncertainty_sum += uncertainty.sum().item()
    return correct / total, uncertainty_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--loss",
        choices=["digamma", "mse"],
        default="digamma",
        help="Evidential loss: Bayes risk of cross-entropy (digamma, default, stable) "
        "or Type-3 expected mean square error (mse).",
    )
    args = parser.parse_args()
    loss_fn = edl_digamma_loss if args.loss == "digamma" else edl_mse_loss

    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device} (Apple Silicon M2) | evidential loss: {args.loss}")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = EDLConvNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"loss": [], "train_acc": [], "test_acc": [], "test_uncertainty": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            target_onehot = F.one_hot(labels, NUM_CLASSES).float()

            alpha = model(images)
            loss = loss_fn(alpha, target_onehot, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            probs, _ = dirichlet_uncertainty(alpha)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        train_acc = correct / total
        test_acc, test_u = evaluate(model, test_loader, device)
        history["loss"].append(epoch_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_uncertainty"].append(test_u)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | loss {epoch_loss:.4f} | "
            f"train acc {train_acc:.4f} | test acc {test_acc:.4f} | "
            f"mean test u {test_u:.3f} | {time.time() - start:.1f}s"
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "epochs": args.epochs,
            "final_test_acc": history["test_acc"][-1],
        },
        "edl_model.pth",
    )
    print("Saved checkpoint -> edl_model.pth")

    epochs = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["loss"], "o-", color="crimson")
    ax1.set_xlabel("Epoch")
    loss_label = "E[CE] (digamma) + KL" if args.loss == "digamma" else "E[MSE] + Var + KL"
    ax1.set_ylabel(f"Evidential loss ({loss_label})")
    ax1.set_title(f"Training Loss (EDL / Dirichlet, {args.loss})")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train accuracy")
    ax2.plot(epochs, history["test_acc"], "s-", label="Test accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Accuracy on FashionMNIST (MPS)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Evidential Deep Learning — Training Metrics", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig("training_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved curves -> training_metrics.png")
    print(f"Final: test acc {history['test_acc'][-1]:.4f}, mean test uncertainty {history['test_uncertainty'][-1]:.3f}")


if __name__ == "__main__":
    main()
