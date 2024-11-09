import logging
import os

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerBase

from ..arguments import ModelArguments

logger = logging.getLogger(__name__)


class SpladeMaxPooling(nn.Module):
    """
    SPLADE Max pooling implementation based on:

    "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval"
    Formal et al., 2021 (https://arxiv.org/abs/2109.10086)

    This implements the max pooling variant introduced in SPLADE v2, which replaced
    the original sum pooling with max pooling over the sequence length dimension:
    w_j = max_{i in t} log(1 + ReLU(w_ij))

    The pooling operation consists of:
    1. Applying ReLU activation
    2. Adding 1 and taking log: log(1 + ReLU(x))
    3. Max pooling over sequence length dimension
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, output, attention_mask):
        """
        Forward pass of SPLADE Max.

        Args:
            output (torch.Tensor): Input tensor of shape (batch_size, sequence_length, vocab_size)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        if output.dim() != 3 or attention_mask.dim() != 2:
            raise ValueError("Invalid input dimensions")

        if output.size(0) != attention_mask.size(0) or output.size(
            1
        ) != attention_mask.size(1):
            raise ValueError("Mismatched batch size or sequence length")

        activated = torch.log(1 + self.relu(output))
        masked = activated * attention_mask.unsqueeze(-1)
        values, _ = torch.max(masked, dim=1)

        return values


class Splade(nn.Module):
    tokenizer: PreTrainedTokenizerBase | None = None

    def __init__(
        self,
        hf_model: PreTrainedModel,
        model_args: ModelArguments,
    ):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.splade_max = SpladeMaxPooling()

        self.config = self.hf_model.config
        self._keys_to_ignore_on_save = getattr(
            self.hf_model, "_keys_to_ignore_on_save", None
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch_inputs: dict, batch_size: int):
        logits = self.hf_model(**batch_inputs, return_dict=True).logits
        logits = self.splade_max(logits, attention_mask=batch_inputs["attention_mask"])
        logits_shape = logits.shape
        assert len(logits_shape) == 2
        query_with_docs_size = int(
            (logits_shape[0] / batch_size)
        )  # query(1) + (group_size = pos(1) + negs(N))
        vocab_size = logits.shape[-1]

        state = logits.view(
            batch_size,
            query_with_docs_size,
            vocab_size,
        )
        queries = state[:, :1, :]
        docs = state[:, 1:, :]

        return queries, docs

    @classmethod
    def from_pretrained(cls, model_args: ModelArguments, *args, **kwargs):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        # for resume training
        model_args_path = os.path.join(
            kwargs.get("pretrained_model_name_or_path", ""), "model_args.bin"
        )
        if os.path.exists(model_args_path):
            model_args = torch.load(model_args_path)
        splade = cls(hf_model, model_args)
        return splade

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
        # for resume training
        torch.save(self.model_args, os.path.join(output_dir, "model_args.bin"))
