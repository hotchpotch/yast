"""
code base from
- https://github.com/FlagOpen/FlagEmbedding/ (MIT license)
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,  # type: ignore
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_subword: bool = field(
        default=False,
        metadata={"help": "Use subword splade model"},
    )


@dataclass
class SpladeTrainingArguments(TrainingArguments):
    sparsity_weight_doc: float = field(
        default=5e-3,
        metadata={
            "help": "Regularization coefficient for document representations. "
            "Controls the sparsity and FLOPS of document embeddings. "
            "Higher values lead to sparser representations."
        },
    )
    sparsity_weight_query: float = field(
        default=5e-3,
        metadata={
            "help": "Regularization coefficient for query representations. "
            "Controls the sparsity and FLOPS of query embeddings. "
            "Higher values lead to sparser representations."
        },
    )

    sparsity_warmup_steps_query: float = field(
        default=0.1,
        metadata={
            "help": "Query lambda warmup steps. If 0 < value < 1, treated as ratio of total steps. Otherwise, absolute step count."
        },
    )
    sparsity_warmup_steps_doc: float = field(
        default=0.1,
        metadata={
            "help": "Document lambda warmup steps. If 0 < value < 1, treated as ratio of total steps. Otherwise, absolute step count."
        },
    )
    regularizer_query: Literal["mean_squared", "flops", "L1", "L2"] = field(
        default="L1",
        metadata={},
    )
    regularizer_doc: Literal["mean_squared", "flops", "L1", "L2"] = field(
        default="flops",
        metadata={},
    )

    training_losses: Any = field(
        default="cross_entropy",
        metadata={
            "help": """Specify single or multiple training losses with optional weights and parameters.
            
    Supported formats:
    1. Single loss (string):
    "cross_entropy"

    2. Multiple losses with weights and parameters (dict):
    {
        "cross_entropy": {
        "weight": 1.0,
        "loss_kwargs": {}
        },
        "mse": {
        "weight": 1.0,
        "loss_kwargs": {}
        }
    }

    Available loss functions: [cross_entropy, mse, contrastive, ...]

    Example configurations:
    - "cross_entropy"  # Single loss
    - {"cross_entropy": {"weight": 1.0}, "mse": {"weight": 0.5}}  # Multiple losses with weights
    - {"cross_entropy": {"weight": 1.0, "loss_kwargs": {"reduction": "mean"}}}  # With parameters
    """
        },
    )
    noise_tokens: None | str | list[str] = field(
        default=None,
        metadata={"help": "Noise tokens for training"},
    )
    noise_tokens_weight: float = field(
        default=1.0,
        metadata={"help": "Noise tokens loss weight"},
    )


@dataclass
class DataArguments:
    train_data: Any = field(
        default=None, metadata={"help": "Path or hf dataset to corpus"}
    )  # type: ignore
    train_group_size: int = field(default=8)
    train_max_positive_size: int = field(default=1)
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input document length after tokenization for input text. "
        },
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input query length after tokenization for input text."
        },
    )
    dataset_options: dict = field(
        default_factory=dict, metadata={"help": "Additional options for the dataset"}
    )

    def __post_init__(self):
        # validation
        pass


@dataclass
class RunArguments:
    remove_checkpoints: bool = field(
        default=False, metadata={"help": "Remove checkpoints after training"}
    )
