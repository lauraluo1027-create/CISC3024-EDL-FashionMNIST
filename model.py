"""Evidential Deep Learning (EDL) model for image classification.

Instead of a softmax point estimate, the CNN outputs non-negative *evidence*
for each class. Following Sensoy et al. (NeurIPS 2018, "Evidential Deep
Learning to Quantify Classification Uncertainty"), the Dirichlet
concentration parameters are alpha_k = e_k + 1. The class probability vector
p is then treated as a random variable p ~ Dir(alpha), which is exactly the
Bayesian view: the network predicts a *distribution over distributions*
rather than a single softmax vector.

    predictive mean:      E[p_k] = alpha_k / S,   S = sum(alpha)
    Bayesian uncertainty: u      = K / S          (vacuity, K classes)
    predictive variance:  Var(p_k) = alpha_k (S - alpha_k) / (S^2 (S + 1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLConvNet(nn.Module):
    """Lightweight CNN whose head parameterizes a Dirichlet distribution."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        logits = self.classifier(self.features(x))
        evidence = F.softplus(logits)  # e_k >= 0
        return evidence + 1.0          # alpha_k >= 1 (Dirichlet concentrations)


def dirichlet_uncertainty(alpha):
    """Predictive mean probabilities and Bayesian vacuity uncertainty u = K/S."""
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    uncertainty = alpha.size(1) / S
    return probs, uncertainty


def kl_divergence(alpha):
    """KL( Dir(alpha) || Dir(1) ): divergence from the uniform (max-entropy) prior."""
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second = (
        (alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)
    return first + second


def edl_mse_loss(alpha, target_onehot, epoch, annealing_step: float = 10.0):
    """Type-3 'Expected Mean Square Error' evidential loss.

    Bayes risk of the squared-error loss under the Dirichlet belief:

        L = sum_k (y_k - E[p_k])^2 + Var(p_k)

    plus a KL regularizer (annealed over epochs) that shrinks evidence for
    the *incorrect* classes toward the uniform prior, so the network only
    carries evidence it can justify.
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    mse = ((target_onehot - p) ** 2).sum(dim=1, keepdim=True)
    var = (p * (1.0 - p) / (S + 1.0)).sum(dim=1, keepdim=True)
    annealing_coef = min(1.0, epoch / annealing_step)
    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence(alpha_tilde)
    return (mse + var + annealing_coef * kl).mean()


def edl_digamma_loss(alpha, target_onehot, epoch, annealing_step: float = 10.0):
    """Bayes risk of the cross-entropy loss under the Dirichlet belief.

        L = sum_k y_k * E[-ln p_k] = sum_k y_k * (digamma(S) - digamma(alpha_k))

    Same evidential objective family as `edl_mse_loss` (expected loss under
    p ~ Dir(alpha), plus the annealed KL prior-shrinkage), but with
    cross-entropy-strength gradients, which train far more stably: the MSE
    variant has ~20x weaker gradients near initialization and a degenerate
    zero-evidence attractor (u -> 1) at larger learning rates.
    """
    S = alpha.sum(dim=1, keepdim=True)
    expected_ce = (target_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1, keepdim=True)
    annealing_coef = min(1.0, epoch / annealing_step)
    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_divergence(alpha_tilde)
    return (expected_ce + annealing_coef * kl).mean()
