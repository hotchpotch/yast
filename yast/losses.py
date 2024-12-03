from typing import Literal, Type, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SPLADE 学習のためのロス実装
"""


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
        # XXX: dict で返せるようにする
        return loss


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature: float = 1.5):
        """
        Args:
            temperature (float): スコアのスケーリングを制御する温度パラメータ
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature

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

        scaled_scores = scores / self.temperature
        log_probs = F.log_softmax(scaled_scores, dim=1)
        teacher_probs = F.softmax(labels / self.temperature, dim=1)
        loss = -(teacher_probs * log_probs).sum(dim=1).mean()
        return loss


class WeightedMarginLoss(nn.Module):
    def __init__(
        self,
        margin: float = 4.0,
        max_negative_teacher: float = 0.3,
        min_positive_teacher: float = 0.7,
    ):
        """
        Args:
            margin (float): positive/negative間の基本マージン
            max_negative_teacher (float): negative例の教師スコアの最大期待値
            min_positive_teacher (float): positive例の教師スコアの最小期待値
        """
        super(WeightedMarginLoss, self).__init__()
        self.base_margin = margin
        self.max_negative_teacher = max_negative_teacher
        self.min_positive_teacher = min_positive_teacher

    def get_margin(self, teacher_scores: torch.Tensor) -> torch.Tensor:
        """
        negative例の教師スコアに応じたマージンを計算
        低いスコアの例（簡単なnegative）ほど大きいマージンを設定

        Args:
            teacher_scores: negative例の教師スコア (batch_size, num_negatives)
        Returns:
            torch.Tensor: マージン値
        """
        # 教師スコアを[0, max_negative_teacher]の範囲にクリップ
        clipped_scores = torch.clamp(teacher_scores, 0.0, self.max_negative_teacher)

        # スコアが低いほどマージンを大きく（0.9-1.1の範囲）
        margin_scale = 1.1 - (clipped_scores / self.max_negative_teacher) * 0.2
        return self.base_margin * margin_scale

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): 内積値 (batch_size, num_candidates)
            labels (torch.Tensor): 教師スコア (batch_size, num_candidates)
                                 最初の列がpositive、残りがnegative
        Returns:
            torch.Tensor: スカラー値のロス
        """
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        # 教師スコアの範囲チェック（警告を出すだけ）
        negatives_mask = labels[:, 1:] > self.max_negative_teacher
        if torch.any(negatives_mask):
            count = torch.sum(negatives_mask).item()
            print(
                f"Warning: {count} negative teacher scores exceed expected maximum value of {self.max_negative_teacher}"
            )

        positives_mask = labels[:, 0] < self.min_positive_teacher
        if torch.any(positives_mask):
            count = torch.sum(positives_mask).item()
            print(
                f"Warning: {count} positive teacher scores are below minimum value of {self.min_positive_teacher}"
            )

        positives = scores[:, 0].unsqueeze(1)  # (batch_size, 1)
        negatives = scores[:, 1:]  # (batch_size, num_negatives)
        teacher_weights = labels[:, 1:]  # negativesの教師スコア

        # マージンの計算
        weighted_margin = self.get_margin(teacher_weights)

        # マージンロスの計算
        margin_diffs = negatives - positives + weighted_margin
        loss = F.relu(margin_diffs).mean()
        return loss


class WeightedMarginLossWithLog(nn.Module):
    def __init__(
        self,
        margin: float = 0.5,
        max_negative_teacher: float = 0.3,
        min_positive_teacher: float = 0.7,
        eps: float = 1e-6,
    ):
        super().__init__()
        """
        margin = 0.5  # これは対数空間での差0.35に対して適切
        positive の内積の統計量は
        mean: 10.03 
          log(10.03) = 2.31
        std: 1.566

        negatives の内積の統計量は
        mean: 7.11
          log(7.11) = 1.96
        std: 1.26
        """
        self.base_margin = margin
        self.max_negative_teacher = max_negative_teacher
        self.min_positive_teacher = min_positive_teacher
        self.eps = eps

    def get_margin(
        self, teacher_pos_scores: torch.Tensor, teacher_neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        positive、negative両方の教師スコアに基づいてマージンを調整

        Args:
            teacher_pos_scores: positive例の教師スコア (batch_size, 1)
            teacher_neg_scores: negative例の教師スコア (batch_size, num_negatives)
        """
        # Positiveスコアの正規化 (0.7-1.0 → 0-1)
        pos_confidence = (teacher_pos_scores - self.min_positive_teacher) / (
            1.0 - self.min_positive_teacher
        )

        # Negativeスコアの正規化 (0-0.3 → 0-1)
        neg_confidence = teacher_neg_scores / self.max_negative_teacher

        # Positiveの確信度が高いほどマージンを大きく (1.0-1.2の範囲)
        pos_scale = 1.0 + (pos_confidence * 0.2)

        # Negativeの確信度が低いほどマージンを大きく (0.9-1.1の範囲)
        neg_scale = 1.1 - (neg_confidence * 0.2)

        # 両方のスケールを組み合わせる
        margin_scale = pos_scale * neg_scale

        return self.base_margin * margin_scale

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        log_scores = torch.log(scores + self.eps)

        positives = log_scores[:, 0].unsqueeze(1)
        negatives = log_scores[:, 1:]

        # 教師スコアを分離
        teacher_pos = labels[:, 0].unsqueeze(1)  # positiveの教師スコア
        teacher_neg = labels[:, 1:]  # negativeの教師スコア

        # 両方の教師スコアを使ってマージンを計算
        weighted_margin = self.get_margin(teacher_pos, teacher_neg)

        margin_diffs = negatives - positives + weighted_margin
        loss = F.relu(margin_diffs).mean()

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
    "soft_ce": SoftCrossEntropyLoss,
    "weighted_margin": WeightedMarginLoss,
    "weighted_margin_log": WeightedMarginLossWithLog,
}
