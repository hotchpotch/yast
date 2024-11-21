from typing import Callable, Concatenate, Literal

import torch


def regularize_flops(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    FLOPs regularization as described in "Minimizing FLOPs to Learn Efficient Sparse Representations".
    https://arxiv.org/abs/2004.05665

    Merits for SPLADE:
    - Directly optimizes for search efficiency by minimizing FLOPs
    - Promotes even distribution of non-zero elements across dimensions
    - Theoretically grounded approach for sparse representation learning

    Demerits for SPLADE:
    - May require careful tuning of regularization strength
    - Might lead to over-sparsification if not balanced with the main loss

    Args:
    batch_tensor (torch.Tensor): Input tensor of shape (batch_size, dim)

    Returns:
    torch.Tensor: FLOPs regularization term
    """
    mean_abs = torch.abs(batch_tensor).mean(dim=0)
    flops_reg = torch.sum(torch.square(mean_abs))
    return flops_reg


def regularize_mean_squared(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Regularization that computes the mean of absolute values, squares them, and sums the result.

    Merits for SPLADE:
    - Similar effect to FLOPs regularization, promoting sparsity
    - May be more numerically stable in some cases

    Demerits for SPLADE:
    - Less direct theoretical connection to FLOPs minimization
    - Might not distribute non-zero elements as evenly as FLOPs regularization

    Args:
    batch_tensor (torch.Tensor): Input tensor of shape (batch_size, dim)

    Returns:
    torch.Tensor: Mean squared regularization term
    """
    return torch.pow(torch.abs(batch_tensor).mean(dim=0), 2).sum()


def regularize_L1(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    L1 regularization: computes the mean L1 norm across the last dimension.

    Merits for SPLADE:
    - Promotes general sparsity in the embeddings
    - Well-understood and widely used in machine learning

    Demerits for SPLADE:
    - Doesn't specifically optimize for search efficiency
    - May not distribute non-zero elements evenly across dimensions

    Args:
    batch_tensor (torch.Tensor): Input tensor of shape (batch_size, dim)

    Returns:
    torch.Tensor: L1 regularization term
    """
    if batch_tensor.size(1) == 1:
        batch_tensor = batch_tensor.squeeze(1)
    return torch.norm(batch_tensor, p=1, dim=-1).mean()


def regularize_L2(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    L2 regularization: computes the mean L2 norm across the last dimension.

    Merits for SPLADE:
    - Prevents embeddings from growing too large
    - Can improve generalization in some cases

    Demerits for SPLADE:
    - Doesn't promote sparsity, which is crucial for SPLADE
    - May not be suitable as the primary regularization for sparse retrieval models

    Args:
    batch_tensor (torch.Tensor): Input tensor of shape (batch_size, dim)

    Returns:
    torch.Tensor: L2 regularization term
    """
    if batch_tensor.size(1) == 1:
        batch_tensor = batch_tensor.squeeze(1)
    l2_norm = torch.norm(batch_tensor, p=2, dim=-1)
    return l2_norm.mean()


def regularize_flops_l1_weighted(
    batch_tensor: torch.Tensor, flops_weight: float = 0.3, l1_weight: float = 0.7
) -> torch.Tensor:
    """
    Combines FLOPs and L1 regularization with adjustable weights.
    Balances between FLOPs' distribution properties and L1's sparsification.

    Merits:
    - Balances between FLOPs' distribution properties and L1's sparsification
    - Adjustable trade-off via flops_weight and l1_weight parameters
    - More flexible than single regularization approaches

    Demerits:
    - Requires tuning of two hyperparameters
    - May be more computationally expensive

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    flops_weight: Weight for FLOPs regularization
    l1_weight: Weight for L1 regularization

    Returns:
    torch.Tensor: Combined regularization term
    """
    flops_term = regularize_flops(batch_tensor)
    l1_term = regularize_L1(batch_tensor)
    return flops_weight * flops_term + l1_weight * l1_term


def regularize_dynamic_sparsity(
    batch_tensor: torch.Tensor, target_sparsity: float = 0.95, smoothing: float = 0.01
) -> torch.Tensor:
    """
    Dynamically adjusts regularization strength based on current sparsity level.

    Merits:
    - Automatically adjusts to maintain desired sparsity level
    - Prevents over-sparsification
    - More stable training dynamics

    Demerits:
    - Additional computational overhead
    - May take longer to converge

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    target_sparsity: Desired sparsity level (0 to 1)
    smoothing: Smoothing factor for sparsity calculation

    Returns:
    torch.Tensor: Adaptive regularization term
    """
    current_sparsity = (batch_tensor.abs() < 1e-6).float().mean()
    sparsity_error = target_sparsity - current_sparsity

    # Adjust regularization strength based on sparsity error
    strength = torch.sigmoid(sparsity_error / smoothing)

    # Use FLOPs regularization with adaptive strength
    return strength * regularize_flops(batch_tensor)


def regularize_magnitude_threshold(
    batch_tensor: torch.Tensor, threshold: float = 0.1, power: float = 2.0
) -> torch.Tensor:
    """
    Applies progressive penalty based on value magnitudes relative to threshold.

    Merits:
    - More granular control over sparsification
    - Can preserve important large values while suppressing small ones
    - Helps achieve desired sparsity pattern

    Demerits:
    - Sensitive to threshold parameter
    - May require careful tuning

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    threshold: Threshold value for penalty application
    power: Power factor for penalty scaling

    Returns:
    torch.Tensor: Threshold regularization term
    """
    abs_values = torch.abs(batch_tensor)
    penalty = torch.where(
        abs_values < threshold,
        torch.pow(abs_values / threshold, power),
        torch.ones_like(abs_values),
    )
    return penalty.mean()


def regularize_entropy_balanced(
    batch_tensor: torch.Tensor, target_entropy: float = 2.0
) -> torch.Tensor:
    """
    Promotes balanced distribution of non-zero elements using entropy.

    Merits:
    - Encourages more balanced distribution of non-zero elements
    - Helps prevent concentration of activations
    - Can maintain semantic diversity

    Demerits:
    - More complex computation
    - May not always converge to optimal sparsity level

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    target_entropy: Target entropy value for distribution

    Returns:
    torch.Tensor: Distributional regularization term
    """
    # Calculate normalized magnitude distribution
    magnitudes = torch.abs(batch_tensor)
    probs = magnitudes / (magnitudes.sum(dim=-1, keepdim=True) + 1e-6)

    # Calculate entropy of the distribution
    entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()

    # Penalize deviation from target entropy
    return torch.abs(entropy - target_entropy)


regularizers: dict[
    Literal[
        "mean_squared",
        "flops",
        "L1",
        "L2",
        "flops_l1_weighted",
        "dynamic_sparsity",
        "magnitude_threshold",
        "entropy_balanced",
    ],
    Callable[Concatenate[torch.Tensor, ...], torch.Tensor],
] = {
    "mean_squared": regularize_mean_squared,
    "flops": regularize_flops,
    "L1": regularize_L1,
    "L2": regularize_L2,
    "flops_l1_weighted": regularize_flops_l1_weighted,
    "dynamic_sparsity": regularize_dynamic_sparsity,
    "magnitude_threshold": regularize_magnitude_threshold,
    "entropy_balanced": regularize_entropy_balanced,
}
