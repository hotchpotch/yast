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
    batch_tensor: torch.Tensor, flops_weight: float = 0.7, l1_weight: float = 0.3
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


def regularize_grouped_magnitude(
    batch_tensor: torch.Tensor, group_size: int = 8, threshold: float = 0.1
) -> torch.Tensor:
    """
    Group-wise magnitude regularization that promotes structured sparsity.
    Similar to structured pruning in neural networks.

    Merits:
    - Promotes structured sparsity within groups
    - More efficient for actual computation
    - Better preserves semantic relationships

    Demerits:
    - Group size needs to be tuned
    - May not work well with very small dimensions

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    group_size: Size of groups for structured sparsity
    threshold: Threshold for magnitude comparison

    Returns:
    torch.Tensor: Group magnitude regularization term
    """
    batch_size, dim = batch_tensor.shape
    num_groups = dim // group_size

    # Reshape into groups
    grouped_tensor = batch_tensor.view(batch_size, num_groups, group_size)

    # Calculate group-wise magnitudes
    group_magnitudes = torch.norm(grouped_tensor, p=2, dim=2)

    # Apply soft thresholding to groups
    threshold_penalty = torch.relu(group_magnitudes - threshold)

    return threshold_penalty.mean()


def regularize_topk_entropy(
    batch_tensor: torch.Tensor, k: int = 256, temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-k sparse entropy regularization that maintains semantic diversity.
    Combines benefits of top-k sparsity with entropy-based distribution control.

    Merits:
    - Controls exact number of non-zero elements
    - Maintains semantic diversity through entropy
    - More predictable sparsity patterns

    Demerits:
    - k needs to be chosen carefully
    - Computationally more expensive due to sorting

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    k: Number of top elements to consider
    temperature: Temperature for softmax

    Returns:
    torch.Tensor: Top-k entropy regularization term
    """
    magnitudes = torch.abs(batch_tensor)

    # Get top-k values and compute soft distribution
    top_k_values, _ = torch.topk(magnitudes, k=k, dim=1)
    soft_distribution = torch.softmax(top_k_values / temperature, dim=1)

    # Compute entropy of top-k distribution
    entropy = -(soft_distribution * torch.log(soft_distribution + 1e-10)).sum(dim=1)

    return -entropy.mean()  # Minimize negative entropy


def regularize_adaptive_threshold(
    batch_tensor: torch.Tensor,
    init_threshold: float = 0.1,
    target_density: float = 0.05,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Adaptive threshold regularization that maintains target density.
    Automatically adjusts threshold to maintain desired sparsity level.

    Merits:
    - Automatically maintains target sparsity
    - Smooth adaptation of threshold
    - More stable than fixed threshold approaches

    Demerits:
    - Requires careful tuning of momentum
    - May take time to stabilize

    Args:
    batch_tensor: Input tensor of shape (batch_size, dim)
    init_threshold: Initial threshold value
    target_density: Target density (1 - sparsity)
    momentum: Momentum for threshold adaptation

    Returns:
    torch.Tensor: Adaptive threshold regularization term
    """
    magnitudes = torch.abs(batch_tensor)
    current_density = (magnitudes > init_threshold).float().mean()

    # Compute threshold adjustment
    density_error = current_density - target_density
    threshold_adjustment = torch.sign(density_error) * torch.abs(density_error).sqrt()

    # Apply soft thresholding with current threshold
    penalty = torch.relu(magnitudes - init_threshold)

    return penalty.mean() * (1.0 + threshold_adjustment)


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
        "dynamic_sparsity",
        "grouped_magnitude",
        "topk_entropy",
        "adaptive_threshold",
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
    "grouped_magnitude": regularize_grouped_magnitude,
    "topk_entropy": regularize_topk_entropy,
    "adaptive_threshold": regularize_adaptive_threshold,
}
