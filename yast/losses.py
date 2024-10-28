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


class TeacherGuidedMarginLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.35,
        margin: float = 2.0,
        soft_ce_weight: float = 1.0,
        margin_weight: float = 1.0,
    ):
        """
        Args:
            temperature (float): スコアのスケーリングを制御する温度パラメータ
            margin (float): positive/negative間の最小マージン
            soft_ce_weight (float): soft cross entropyロスの重み
            margin_weight (float): マージンロスの重み
        """
        super(TeacherGuidedMarginLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.soft_ce_weight = soft_ce_weight
        self.margin_weight = margin_weight

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): 内積値 (batch_size, num_candidates)
            labels (torch.Tensor): 教師スコア (batch_size, num_candidates)
        Returns:
            torch.Tensor: スカラー値のロス
        """
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        # スコアを温度でスケーリング
        scaled_scores = scores / self.temperature

        # Soft Cross Entropy Loss (クロスエントロピー)
        log_probs = F.log_softmax(scaled_scores, dim=1)
        teacher_probs = F.softmax(labels / self.temperature, dim=1)
        soft_ce_loss = -(teacher_probs * log_probs).sum(dim=1).mean()

        # Weighted Margin Loss
        positives = scores[:, 0].unsqueeze(1)  # (batch_size, 1)
        negatives = scores[:, 1:]  # (batch_size, num_negatives)
        teacher_weights = labels[:, 1:]  # negativesの教師スコア
        weighted_margin = self.margin * (
            1.0 - teacher_weights
        )  # (batch_size, num_negatives)
        margin_diffs = (
            negatives - positives + weighted_margin
        )  # (batch_size, num_negatives)
        margin_loss = F.relu(margin_diffs).mean()

        # 最終的なロス（重み付き合計）
        loss = self.soft_ce_weight * soft_ce_loss + self.margin_weight * margin_loss
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
    "teacher_guided_margin": TeacherGuidedMarginLoss,
}
