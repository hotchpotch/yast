import logging
import os

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerBase

from ..arguments import ModelArguments
from .splade import Splade

logger = logging.getLogger(__name__)


class SpladeSubword(Splade):
    SUBWORD_MASK_ID = -100

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizerBase):
        # override
        self._tokenizer = value
        subword_token_ids = []
        for token in value.get_vocab():
            if token.startswith(self.subword_prefix):
                subword_token_ids.append(value.convert_tokens_to_ids(token))  # type: ignore
        self.subword_token_ids = set(subword_token_ids)

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        """
        Args:
            pooling_type (str): プーリングの種類 ('max' または 'mean')
        """
        super().__init__(hf_model, model_args)
        self.relu = nn.ReLU()
        pooling_type = "max"
        self.subword_prefix = "##"
        self.subword_token_ids = set()
        self.pooling_type = pooling_type

    def subword_pooling(self, tensor, subword_indices, attention_mask):
        batch_size, seq_len, vocab_size = tensor.size()
        device = tensor.device

        # 勾配を保持するため、clone()を使用
        masked_tensor = tensor * attention_mask.unsqueeze(-1).to(tensor.dtype)
        pooled_output = torch.zeros_like(tensor)  # requires_gradは継承される

        # 最大グループIDの取得を修正
        valid_indices = subword_indices[subword_indices != self.SUBWORD_MASK_ID]
        if len(valid_indices) == 0:
            return pooled_output

        # intとして最大値を取得
        max_group_id = int(valid_indices.max().item())

        # バッチ処理を行列演算に変更して効率化
        for group_id in range(max_group_id + 1):
            # グループマスクを全バッチで一度に作成
            group_mask = (subword_indices == group_id).float().unsqueeze(-1)
            attention_mask_expanded = attention_mask.unsqueeze(-1).to(tensor.dtype)
            group_mask = group_mask * attention_mask_expanded

            if self.pooling_type == "max":
                # マスク適用と最大値計算を行列演算で実行
                masked_values = masked_tensor * group_mask
                masked_values = masked_values + (-1e9 * (1 - group_mask))
                pooled_values = torch.max(masked_values, dim=1, keepdim=True)[0]
                # 結果を展開
                pooled_output = pooled_output + (pooled_values * group_mask)
            else:  # mean
                sum_values = torch.sum(masked_tensor * group_mask, dim=1, keepdim=True)
                count = torch.sum(group_mask, dim=1, keepdim=True).clamp(min=1e-9)
                pooled_values = sum_values / count
                pooled_output = pooled_output + (pooled_values * group_mask)

        return pooled_output

    def forward(self, batch_inputs: dict, batch_size: int):
        logits = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

        subword_indices = create_subword_indices(
            logits,
            self.subword_token_ids,
        )

        pooled = self.subword_pooling(logits, subword_indices, attention_mask)
        return self._forward_logits(pooled, attention_mask, batch_size)


def create_subword_indices(
    token_ids: torch.Tensor, subword_token_ids: set
) -> torch.Tensor:
    """
    トークンIDからサブワードインデックスを生成する
    サブワードを含む単語は同じインデックスでグループ化し、
    単独トークンは-1として扱う

    Args:
        token_ids (torch.Tensor): トークンID (batch_size, seq_len)
        subword_token_ids (set): サブワードとして扱うトークンIDのset

    Returns:
        torch.Tensor: サブワードインデックス (batch_size, seq_len)
            -1: 単独トークン（サブワードを含まない単語）
            0以上: サブワードを含む単語のグループインデックス
    """
    batch_size, seq_len = token_ids.shape
    subword_indices = torch.full_like(
        token_ids,
        -100,  # PADDINGのマスク値
    )

    for b in range(batch_size):
        current_word_idx = -1
        word_start_pos = -1
        in_subword_sequence = False

        for i in range(seq_len):
            token_id = token_ids[b, i].item()

            # パディングやマスクされたトークンはスキップ
            if token_id == -100:
                continue

            is_subword = token_id in subword_token_ids

            # 新しい単語の開始
            if not is_subword and not in_subword_sequence:
                # 前の単語の処理
                if word_start_pos != -1:
                    # 単独トークンの場合
                    if not in_subword_sequence:
                        subword_indices[b, word_start_pos] = -100

                word_start_pos = i
                in_subword_sequence = False

            # サブワードシーケンスの開始
            elif is_subword and not in_subword_sequence:
                current_word_idx += 1
                in_subword_sequence = True
                # 直前のトークンも同じグループに
                if word_start_pos != -1:
                    subword_indices[b, word_start_pos : i + 1] = current_word_idx

            # サブワードシーケンスの途中
            elif is_subword and in_subword_sequence:
                subword_indices[b, i] = current_word_idx

            # サブワードシーケンスの終了
            if not is_subword and in_subword_sequence:
                word_start_pos = i
                in_subword_sequence = False

        # 最後の単語の処理
        if word_start_pos != -1 and not in_subword_sequence:
            subword_indices[b, word_start_pos] = -100

    return subword_indices
