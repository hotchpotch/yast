import logging
import os
import re
from typing import Any, Union

import torch
from transformers import Trainer

from .arguments import SpladeTrainingArguments
from .log_metrics import LogMetrics
from .losses import LossWithWeight, losses
from .modeling import Splade
from .regularizers import regularizers

logger = logging.getLogger(__name__)


class SpladeTrainer(Trainer):
    def __init__(self, args: SpladeTrainingArguments, **kwargs: Any):
        super().__init__(args=args, **kwargs)
        self.args: SpladeTrainingArguments = args
        self.batch_size: int = self.args.per_device_train_batch_size

        self.total_steps: int = int(
            self.get_train_dataloader().__len__() * args.num_train_epochs
        )

        self.warmup_steps_doc: int = self._calculate_warmup_steps(
            args.sparsity_warmup_steps_doc
        )
        self.warmup_steps_query: int = self._calculate_warmup_steps(
            args.sparsity_warmup_steps_query
        )

        self._set_training_losses(args.training_losses)
        self._set_noise_token_ids(args.noise_tokens)

        self.regularizer_query_fn = regularizers[args.regularizer_query]
        self.regularizer_doc_fn = regularizers[args.regularizer_doc]

        self.log_metrics: LogMetrics = LogMetrics()

    def _set_noise_token_ids(self, noise_tokens: None | str | list[str]) -> None:
        if isinstance(noise_tokens, str):
            noise_tokens = re.split(r"\s+", noise_tokens)
        elif noise_tokens is None:
            noise_tokens = []
        if len(noise_tokens) == 0:
            self.noise_token_ids = []
        else:
            noise_tokens = list(set(noise_tokens))
            tokenizer = self.tokenizer
            token_ids: list[int] = tokenizer.convert_tokens_to_ids(noise_tokens)  # type: ignore
            if len(token_ids) != len(noise_tokens):
                missing_tokens = set(noise_tokens) - set(
                    tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
                )
                raise ValueError(
                    f"Token(s) {missing_tokens} are not in the tokenizer's vocabulary."
                )
            logger.info(
                f"target noise tokens ({len(token_ids)}): {' '.join(noise_tokens)}"
            )
            self.noise_token_ids = token_ids

    def _set_training_losses(self, training_loss: Any) -> None:
        """
        'cross_entropy',  [['loss': 'cross_entropy', 'weight': 1.0], ['loss': 'mse', 'weight': 1.0]]"
        training_loss types
        1) str only
          ex) 'cross_entropy'
        2) dict
          ex) {"cross_entropy": {"weight": 1.0", loss_kwargs: {}}, "mse": {"weight": 1.0, loss_kwargs: {}}}
        """

        training_losses: dict[str, LossWithWeight] = {}
        if isinstance(training_loss, str):
            if loss_klass := losses.get(training_loss):
                loss_fn = loss_klass()
                loss_with_args: LossWithWeight = {
                    "loss_fn": loss_fn,
                    "weight": 1.0,
                }
                training_losses[training_loss] = loss_with_args
            else:
                raise ValueError(
                    f"Training loss type {training_loss} is not supported. Choose from {list(losses.keys())}"
                )
        elif isinstance(training_loss, dict):
            for loss_name, loss_values in training_loss.items():
                if loss_klass := losses.get(loss_name):
                    loss_fn = loss_klass(**loss_values.get("loss_kwargs", {}))
                    loss_with_args: LossWithWeight = {
                        "loss_fn": loss_fn,
                        "weight": loss_values.get("weight", 1.0),
                    }
                    training_losses[loss_name] = loss_with_args
                else:
                    raise ValueError(
                        f"Training loss type {loss_name} is not supported. Choose from {list(losses.keys())}"
                    )
        else:
            raise ValueError(
                f"Training loss type {training_loss} is not supported. Choose from {list(losses.keys())}"
            )
        self.training_losses = training_losses
        self.training_loss_is_contrastive = (
            "contrastive" in training_losses and len(training_losses) == 1
        )

    def _calculate_warmup_steps(self, steps: Union[float, int]) -> int:
        if 0.0 < steps < 1.0:
            return int(self.total_steps * steps)
        return int(steps)

    def _calculate_current_warmup_weight(
        self, max_weight: float, warmup_steps: int
    ) -> float:
        step = self.state.global_step
        current_weight = max_weight * ((step) / (warmup_steps + 1)) ** 2
        return min(max_weight, current_weight)

    @property
    def current_sparsity_weight_doc(self) -> float:
        return self._calculate_current_warmup_weight(
            self.args.sparsity_weight_doc, self.warmup_steps_doc
        )

    @property
    def current_sparsity_weight_query(self) -> float:
        return self._calculate_current_warmup_weight(
            self.args.sparsity_weight_query, self.warmup_steps_query
        )

    def compute_loss(
        self,
        model: Splade,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,  # for transformers v4.46.0
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if "labels" in inputs:
            labels = inputs.pop("labels")
            labels = labels[~torch.isnan(labels)]  # remove nan values
            labels = labels.view(self.args.per_device_train_batch_size, -1)
        else:
            raise ValueError("labels is not in batch_inputs")

        queries, docs = model(inputs, self.batch_size)
        vocab_size = docs.size(2)

        scores = self.compute_scores(queries, docs)
        losses = {}
        for loss_name, loss_with_args in self.training_losses.items():
            loss_fn = loss_with_args["loss_fn"]
            weight = loss_with_args["weight"]
            loss = loss_fn(scores, labels)
            losses[loss_name + "_loss"] = weight * loss

        loss = sum(losses.values())

        docs_matrix = docs.reshape(-1, vocab_size)
        queries_matrix = queries.reshape(-1, vocab_size)

        regularizer_doc_loss = (
            self.regularizer_doc_fn(docs_matrix) * self.current_sparsity_weight_doc
        )
        regularizer_query_loss = (
            self.regularizer_query_fn(queries_matrix)
            * self.current_sparsity_weight_query
        )
        regularizer_loss = regularizer_doc_loss + regularizer_query_loss
        noise_token_loss = self.compute_noise_token_loss(queries_matrix, docs_matrix)

        losses: dict[str, float | torch.Tensor] = {
            **losses,
            "L0_doc": LogMetrics.L0(docs_matrix),
            "L0_query": LogMetrics.L0(queries_matrix),
            "regularizer_loss": regularizer_loss,
            "regularizer_doc_loss": regularizer_doc_loss,
            "regularizer_query_loss": regularizer_query_loss,
            "noise_token_loss": noise_token_loss,
        }
        self.log_metrics.add_dict(losses)

        total_loss = loss + regularizer_loss + noise_token_loss

        if not return_outputs:
            return total_loss
        return total_loss, [(queries, docs)]

    def compute_noise_token_loss(
        self, queries_matrix: torch.Tensor, docs_matrix: torch.Tensor
    ) -> torch.Tensor:
        if self.noise_token_ids and self.args.noise_tokens_weight > 0:
            noise_token_ids_tensor = torch.tensor(
                self.noise_token_ids, device=queries_matrix.device
            )
            noise_token_ids_tensor = noise_token_ids_tensor.view(-1)

            noise_scores_queries = queries_matrix[:, noise_token_ids_tensor]
            noise_scores_docs = docs_matrix[:, noise_token_ids_tensor]

            noise_loss_queries = noise_scores_queries.sum()
            noise_loss_docs = noise_scores_docs.sum()
            noise_token_loss = noise_loss_queries + noise_loss_docs

            # warmup
            current_weight = self._calculate_current_warmup_weight(
                self.args.noise_tokens_weight, self.warmup_steps_query
            )
            noise_token_loss *= current_weight
        else:
            noise_token_loss = torch.tensor(0.0, device=queries_matrix.device)
        return noise_token_loss

    def compute_scores(self, queries, docs) -> torch.Tensor:
        if self.training_loss_is_contrastive:
            scores = self.compute_contrastive_score(queries, docs)
        else:
            scores = self.compute_similarity_scores(queries, docs)
        return scores

    def compute_contrastive_score(
        self, queries: torch.Tensor, docs: torch.Tensor
    ) -> torch.Tensor:
        """
        for contrastive loss,
        return shape is (batch_size, 1 + neg_size)
        """
        scores = torch.bmm(queries, torch.permute(docs, [0, 2, 1])).squeeze(1)
        scores_positive = scores[:, :1]
        negatives = docs[:, 1:, :].reshape(-1, docs.size(2)).T
        scores_negative = torch.matmul(queries.squeeze(1), negatives)
        return torch.cat([scores_positive, scores_negative], dim=1)

    def compute_similarity_scores(
        self, queries: torch.Tensor, docs: torch.Tensor
    ) -> torch.Tensor:
        return torch.bmm(queries, docs.transpose(1, 2)).squeeze(1)

    def log(self, logs: dict[str, float]) -> None:
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        current_metrics = self.log_metrics.mean()
        self.log_metrics.clear()
        logs.update(current_metrics)

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def _save(
        self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None
    ) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, "save_pretrained"):
            raise NotImplementedError(
                f"MODEL {self.model.__class__.__name__} does not support save_pretrained interface"
            )

        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
