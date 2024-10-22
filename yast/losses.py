from typing import Literal, Type, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(
        self,
        reduction: Literal["batchmean", "sum", "none"] = "batchmean",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )
        student_log_probs = F.log_softmax(scores / self.temperature, dim=1)
        loss = self.kl_div(student_log_probs, labels) * (self.temperature**2)
        return loss


class MarginMSELoss(nn.Module):
    def __init__(self, margin: float = 0.05):
        super(MarginMSELoss, self).__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        mse_loss = F.mse_loss(scores, labels, reduction="none")
        margin_loss = F.relu(mse_loss - self.margin)

        loss = margin_loss.mean(dim=1).mean()

        return loss


class MarginCrossEntropyLoss(nn.Module):
    def __init__(self, margin: float = 0.05):
        super(MarginCrossEntropyLoss, self).__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        log_probs = F.log_softmax(scores, dim=1)
        soft_ce_loss = -(labels * log_probs).sum(dim=1).mean()

        mse_loss = F.mse_loss(scores, labels, reduction="none").mean(dim=1)
        margin_loss = F.relu(mse_loss - self.margin).mean()

        loss = soft_ce_loss + margin_loss
        return loss


class LossWithWeight(TypedDict):
    loss_fn: nn.Module
    weight: float


losses: dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "kl_div": KLDivLoss,
    "margin_mse": MarginMSELoss,
    "margin_ce": MarginCrossEntropyLoss,
}
