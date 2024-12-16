import logging

import torch
from transformers import PreTrainedModel

from ..arguments import ModelArguments
from .splade import Splade

logger = logging.getLogger(__name__)


class SpladeSubword(Splade):
    POOLING_TYPES = ["max", "mean"]  # Available pooling types
    SUBWORD_MASK_ID = -100  # ID to ignore subword positions

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        super().__init__(hf_model, model_args)
        subword_pooling = model_args.subword_pooling
        if subword_pooling is not None and subword_pooling not in self.POOLING_TYPES:
            raise ValueError(
                f"Invalid pooling type: {subword_pooling}. Please choose from {self.POOLING_TYPES}"
            )
        self.pooling_type = subword_pooling

    def _aggregate_subwords(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        subword_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized aggregation of subword logits using PyTorch's scatter operations.

        Args:
            logits (torch.Tensor): [batch_size, vocab_size] logits tensor after splade_max.
            input_ids (torch.Tensor): [batch_size, sequence_length] input token IDs.
            subword_indices (torch.Tensor): [batch_size, sequence_length] subword group indices.

        Returns:
            torch.Tensor: Aggregated logits tensor.
        """
        batch_size, vocab_size = logits.size()
        device = logits.device
        dtype = logits.dtype  # Ensure consistent dtype

        # Create mask to identify valid subword positions
        mask = subword_indices != self.SUBWORD_MASK_ID  # [B, S]

        # Get indices of valid positions
        valid_positions = mask.nonzero(
            as_tuple=False
        )  # [N, 2] each row is (batch_idx, seq_idx)
        if valid_positions.numel() == 0:
            return logits  # No valid subwords to aggregate

        batch_indices = valid_positions[:, 0]  # [N]
        seq_indices = valid_positions[:, 1]  # [N]

        # Extract corresponding group indices and token IDs
        group_indices = subword_indices[batch_indices, seq_indices]  # [N]
        token_ids = input_ids[batch_indices, seq_indices]  # [N]

        # Get logits corresponding to selected token IDs
        selected_logits = logits[batch_indices, token_ids]  # [N]

        # Calculate unique group identifiers by offsetting group IDs for each batch
        max_group_tensor = group_indices.max()
        max_group = (
            max_group_tensor.item()
            if torch.is_tensor(max_group_tensor)
            else max_group_tensor
        )
        if max_group < 0:
            max_group = 0  # Handle case where all group_indices are -100, though unlikely due to mask

        # Add batch offset to make group identifiers unique
        group_offset = batch_indices * (max_group + 1)  # [N]
        unique_group_ids = group_offset + group_indices  # [N]

        # Calculate number of unique groups
        num_unique_groups = batch_size * (max_group + 1)  # Removed .item()

        if self.pooling_type == "max":
            # Initialize with -inf and set the same dtype as logits
            pooled_values = torch.full(
                (num_unique_groups,),  # type: ignore
                -float("inf"),
                device=device,
                dtype=dtype,  # type: ignore
            )
            # Calculate maximum value for each group using scatter_reduce
            pooled_values = pooled_values.scatter_reduce(
                dim=0,
                index=unique_group_ids,
                src=selected_logits,
                reduce="amax",
                include_self=True,
            )
        elif self.pooling_type == "mean":
            # Initialize sum and count tensors with the same dtype as logits
            sum_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)  # type: ignore
            count_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)  # type: ignore
            # Calculate sum for each group
            sum_pooled = sum_pooled.scatter_add(0, unique_group_ids, selected_logits)
            # Calculate count for each group
            count_pooled = count_pooled.scatter_add(
                0, unique_group_ids, torch.ones_like(selected_logits)
            )
            # Calculate average (prevent division by zero)
            pooled_values = sum_pooled / torch.clamp(count_pooled, min=1)

        # Map pooled values to each token
        pooled_values_per_token = pooled_values[unique_group_ids]  # [N] # type: ignore

        # Calculate global token IDs to handle multiple tokens in batch
        global_token_ids = batch_indices * vocab_size + token_ids  # [N]

        # Initialize tensor to hold maximum pooled values for each token (initialize with -inf)
        final_pooled = torch.full(
            (batch_size * vocab_size,), -float("inf"), device=device, dtype=dtype
        )

        if self.pooling_type == "max":
            # For MaxPooling, use maximum values
            final_pooled = final_pooled.scatter_reduce(
                dim=0,
                index=global_token_ids,
                src=pooled_values_per_token,
                reduce="amax",
                include_self=True,
            )
            final_pooled = final_pooled.view(batch_size, vocab_size)
            # Take maximum between original logits and pooled values
            new_logits = torch.maximum(logits, final_pooled)
        else:  # mean
            # For MeanPooling, use average values directly
            temp_sum = torch.zeros_like(final_pooled)
            temp_count = torch.zeros_like(final_pooled)

            # Aggregate sum of values and count
            temp_sum.scatter_add_(0, global_token_ids, pooled_values_per_token)
            temp_count.scatter_add_(
                0, global_token_ids, torch.ones_like(pooled_values_per_token)
            )

            # Calculate final average
            final_pooled = (temp_sum / temp_count.clamp(min=1.0)).view(
                batch_size, vocab_size
            )

            # Update only positions with subwords using average values
            subword_mask = (temp_count > 0).view(batch_size, vocab_size)
            new_logits = torch.where(subword_mask, final_pooled, logits)

        return new_logits

    def forward(self, batch_inputs: dict, batch_size: int):
        """
        Forward pass of the model.

        Args:
            batch_inputs: Input batch containing 'input_ids', 'attention_mask', and 'subword_indices'.
            batch_size: Size of the batch.

        Returns:
            Tuple of query and document representations.
        """
        subword_indices = batch_inputs.pop("subword_indices")
        input_ids = batch_inputs["input_ids"]

        output = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

        logits = self.splade_max(output, attention_mask)

        if self.pooling_type is not None:
            logits = self._aggregate_subwords(logits, input_ids, subword_indices)

        # Get query and document representations
        query, docs = self._logit_to_query_docs(logits, batch_size)
        return query, docs
