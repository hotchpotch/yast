from typing import Callable, Literal

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


regularizers: dict[
    Literal["mean_squared", "flops", "L1", "L2"], Callable[[torch.Tensor], torch.Tensor]
] = {
    "mean_squared": regularize_mean_squared,
    "flops": regularize_flops,
    "L1": regularize_L1,
    "L2": regularize_L2,
}
