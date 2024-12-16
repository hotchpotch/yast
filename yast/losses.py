from typing import Literal, Type, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Loss implementations for SPLADE training
"""


class KLDivLoss(nn.Module):
    def __init__(
        self,
        reduction: Literal["batchmean", "sum", "none"] = "batchmean",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        if not torch.isfinite(scores).all():
            raise ValueError("scores contains inf or nan")

        if not torch.isfinite(labels).all():
            raise ValueError("labels contains inf or nan")

        log_probs = F.log_softmax(scores / self.temperature, dim=1)
        loss = self.kl_div(log_probs, labels) * (self.temperature**2)

        return loss


class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        temperature: float = 1.0,
        scaling_factor: float = 25.0,
        pos_weight: float = 8.0,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if scaling_factor <= 0:
            raise ValueError(f"Scaling factor must be positive, got {scaling_factor}")

        self.temperature = temperature
        self.scaling_factor = scaling_factor
        # Initialize BCEWithLogitsLoss with pos_weight
        self.bce = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        if not torch.isfinite(scores).all():
            raise ValueError("scores contains inf or nan")

        if not torch.isfinite(labels).all():
            raise ValueError("labels contains inf or nan")

        scaled_scores = (scores / self.scaling_factor) / self.temperature
        loss = self.bce(scaled_scores, labels) * (self.temperature**2)

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
            temperature (float): Temperature parameter controlling score scaling
            margin (float): Minimum margin between positive/negative pairs
            soft_ce_weight (float): Weight for soft cross entropy loss
            margin_weight (float): Weight for margin loss
        """
        super(TeacherGuidedMarginLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.soft_ce_weight = soft_ce_weight
        self.margin_weight = margin_weight

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): Inner product values (batch_size, num_candidates)
            labels (torch.Tensor): Teacher scores (batch_size, num_candidates)
        Returns:
            torch.Tensor: Scalar loss value
        """
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        # Scale scores by temperature
        scaled_scores = scores / self.temperature

        # Soft Cross Entropy Loss
        log_probs = F.log_softmax(scaled_scores, dim=1)
        teacher_probs = F.softmax(labels / self.temperature, dim=1)
        soft_ce_loss = -(teacher_probs * log_probs).sum(dim=1).mean()

        # Weighted Margin Loss
        positives = scores[:, 0].unsqueeze(1)  # (batch_size, 1)
        negatives = scores[:, 1:]  # (batch_size, num_negatives)
        teacher_weights = labels[:, 1:]  # Teacher scores for negatives
        weighted_margin = self.margin * (
            1.0 - teacher_weights
        )  # (batch_size, num_negatives)
        margin_diffs = (
            negatives - positives + weighted_margin
        )  # (batch_size, num_negatives)
        margin_loss = F.relu(margin_diffs).mean()

        # Final loss (weighted sum)
        loss = self.soft_ce_weight * soft_ce_loss + self.margin_weight * margin_loss
        # TODO: Enable returning as dict
        return loss


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature: float = 1.5):
        """
        Args:
            temperature (float): Temperature parameter controlling score scaling
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): Inner product values (batch_size, num_candidates)
            labels (torch.Tensor): Teacher scores (batch_size, num_candidates)
        Returns:
            torch.Tensor: Scalar loss value
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
            margin (float): Base margin between positive/negative pairs
            max_negative_teacher (float): Maximum expected value for negative teacher scores
            min_positive_teacher (float): Minimum expected value for positive teacher scores
        """
        super(WeightedMarginLoss, self).__init__()
        self.base_margin = margin
        self.max_negative_teacher = max_negative_teacher
        self.min_positive_teacher = min_positive_teacher

    def get_margin(self, teacher_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate margin based on negative teacher scores
        Lower scoring examples (easy negatives) get larger margins

        Args:
            teacher_scores: Teacher scores for negative examples (batch_size, num_negatives)
        Returns:
            torch.Tensor: Margin values
        """
        # Clip teacher scores to [0, max_negative_teacher] range
        clipped_scores = torch.clamp(teacher_scores, 0.0, self.max_negative_teacher)

        # Lower scores get larger margins (range 0.9-1.1)
        margin_scale = 1.1 - (clipped_scores / self.max_negative_teacher) * 0.2
        return self.base_margin * margin_scale

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): Inner product values (batch_size, num_candidates)
            labels (torch.Tensor): Teacher scores (batch_size, num_candidates)
                                 First column is positive, rest are negative
        Returns:
            torch.Tensor: Scalar loss value
        """
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        # Check teacher score ranges (warning only)
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
        teacher_weights = labels[:, 1:]  # Teacher scores for negatives

        # Calculate margins
        weighted_margin = self.get_margin(teacher_weights)

        # Calculate margin loss
        margin_diffs = negatives - positives + weighted_margin
        loss = F.relu(margin_diffs).mean()
        return loss


class WeightedMarginLossWithLog(nn.Module):
    def __init__(
        self,
        margin: float = 0.5,
        # max_negative_teacher: float = 0.3,
        # min_positive_teacher: float = 0.7,
        eps: float = 1e-6,
    ):
        super().__init__()
        """
        margin = 0.5  # This is appropriate for a log-space difference of 0.35
        Statistics for positive inner products:
        mean: 10.03 
          log(10.03) = 2.31
        std: 1.566

        Statistics for negative inner products:
        mean: 7.11
          log(7.11) = 1.96
        std: 1.26
        """
        self.base_margin = margin
        # self.max_negative_teacher = max_negative_teacher
        # self.min_positive_teacher = min_positive_teacher
        self.eps = eps

    def get_margin(
        self, teacher_pos_scores: torch.Tensor, teacher_neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjust margins for each batch element (row) based on both positive and negative teacher scores

        Args:
            teacher_pos_scores: Teacher scores for positive examples (batch_size, 1)
            teacher_neg_scores: Teacher scores for negative examples (batch_size, num_negatives)
        Returns:
            torch.Tensor: Adjusted margins (batch_size, num_negatives)
        """
        # Calculate statistics for each row
        neg_mean_per_row = teacher_neg_scores.mean(
            dim=1, keepdim=True
        )  # (batch_size, 1)
        neg_std_per_row = teacher_neg_scores.std(dim=1, keepdim=True)  # (batch_size, 1)

        # Calculate relative position of negative scores in each row
        # How far from the mean (in standard deviations)
        neg_relative_scores = (teacher_neg_scores - neg_mean_per_row) / (
            neg_std_per_row + self.eps
        )

        # Relative strength of positive scores per row
        # Confidence in positive examples for that query
        pos_relative_strength = (teacher_pos_scores - self.min_positive_teacher) / (
            1.0 - self.min_positive_teacher
        )  # (batch_size, 1)

        # Higher positive confidence leads to larger margins (range 1.0-1.2)
        pos_scale = 1.0 + (pos_relative_strength * 0.2)  # (batch_size, 1)

        # Lower negative scores relative to mean get larger margins
        # Map -2~2σ range to 0-1 using sigmoid
        neg_confidence = torch.sigmoid(neg_relative_scores)
        neg_scale = 1.1 - (neg_confidence * 0.2)  # (batch_size, num_negatives)

        # Combine both scales
        margin_scale = pos_scale * neg_scale  # (batch_size, num_negatives)

        return self.base_margin * margin_scale

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} != labels {labels.shape}"
            )

        log_scores = torch.log(scores + self.eps)

        positives = log_scores[:, 0].unsqueeze(1)
        negatives = log_scores[:, 1:]

        # Separate teacher scores
        teacher_pos = labels[:, 0].unsqueeze(1)  # Teacher scores for positives
        teacher_neg = labels[:, 1:]  # Teacher scores for negatives

        # Calculate margins using both teacher scores
        weighted_margin = self.get_margin(teacher_pos, teacher_neg)

        margin_diffs = negatives - positives + weighted_margin
        loss = F.relu(margin_diffs).mean()

        return loss


class LossWithWeight(TypedDict):
    loss_fn: nn.Module
    weight: float


losses: dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "kl_div": KLDivLoss,
    "margin_mse": MarginMSELoss,
    "margin_ce": MarginCrossEntropyLoss,
    "teacher_guided_margin": TeacherGuidedMarginLoss,
    "soft_ce": SoftCrossEntropyLoss,
    "weighted_margin": WeightedMarginLoss,
    "weighted_margin_log": WeightedMarginLossWithLog,
    "weighted_bce": WeightedBCELoss,
}
