import logging

import torch
from torch import nn
from transformers import PreTrainedModel

from ..arguments import ModelArguments
from .splade import Splade

logger = logging.getLogger(__name__)


class SpladeSubword(Splade):
    POOLING_TYPES = ["max", "mean"]
    SUBWORD_MASK_ID = -100

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
        グループごとにlogitsを集約し、新しいlogitsを生成します。
        各グループ内のトークンIDに対してmeanまたはmaxを計算し、
        既存のlogitsと比較して高い値を保持します。

        Args:
            logits (torch.Tensor): [batch_size, vocab_size] の形状のlogitsテンソル。
            input_ids (torch.Tensor): [batch_size, sequence_length] の形状の入力トークンID。
            subword_indices (torch.Tensor): [batch_size, sequence_length] の形状のサブワードインデックス。

        Returns:
            torch.Tensor: 集約後のlogitsテンソル。
        """
        batch_size, vocab_size = logits.size()
        # logitsをコピーして新しいテンソルを作成
        new_logits = logits.clone()

        for b in range(batch_size):
            # 現在のバッチのsubword_indicesを取得
            current_subword_indices = subword_indices[b]
            current_input_ids = input_ids[b]

            # -100を除いたユニークなグループインデックスを取得
            unique_groups = torch.unique(current_subword_indices)
            unique_groups = unique_groups[unique_groups != self.SUBWORD_MASK_ID]

            for group in unique_groups:
                # 現在のグループに属するトークンの位置を取得
                positions = (current_subword_indices == group).nonzero(as_tuple=True)[0]
                if positions.numel() == 0:
                    continue  # グループに属するトークンがない場合はスキップ

                # グループ内のトークンIDを取得し、ユニークにする
                tokens = current_input_ids[positions].unique()

                # これらのトークンIDに対応するlogitsを取得
                token_logits = logits[b, tokens]

                # プーリングタイプに応じて集約値を計算
                if self.pooling_type == "mean":
                    pooled_value = token_logits.mean()
                    # 平均値を設定し、他のグループの影響を考慮して最大値を保持
                    new_logits[b, tokens] = torch.max(
                        new_logits[b, tokens], pooled_value
                    )
                elif self.pooling_type == "max":
                    pooled_value = token_logits.max()
                    # 最大値を設定し、他のグループの影響を考慮して最大値を保持
                    new_logits[b, tokens] = torch.max(
                        new_logits[b, tokens], pooled_value
                    )
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return new_logits

    def forward(self, batch_inputs: dict, batch_size: int):
        """
        モデルのforward処理

        Args:
            batch_inputs: 入力バッチ
            batch_size: バッチサイズ

        Returns:
            モデルの出力
        """
        subword_indices = batch_inputs.pop("subword_indices")
        input_ids = batch_inputs["input_ids"]
        output = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]
        logits = self.splade_max(output, attention_mask)

        # サブワードの集約処理を実行
        if self.pooling_type is not None:
            logits = self._aggregate_subwords(logits, input_ids, subword_indices)

        query, docs = self._logit_to_query_docs(logits, batch_size)
        return query, docs
