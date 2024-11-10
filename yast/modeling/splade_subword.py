import logging

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
        self._tokenizer = value
        subword_token_ids = []
        for token in value.get_vocab():
            if token.startswith(self.subword_prefix):
                subword_token_ids.append(value.convert_tokens_to_ids(token))  # type: ignore
        self.subword_token_ids = set(subword_token_ids)

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        super().__init__(hf_model, model_args)
        self.relu = nn.ReLU()
        self.subword_prefix = "##"
        self.subword_token_ids = set()
        self.pooling_type = "max"

    def subword_pooling(self, tensor, subword_indices, attention_mask):
        """
        サブワードグループごとにプーリングを行う

        Args:
            tensor: 入力テンソル (batch_size, seq_len, vocab_size)
            subword_indices: サブワードグループのインデックス (batch_size, seq_len)
            attention_mask: アテンションマスク (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = tensor.size()
        device = tensor.device

        # 勾配を保持するため、新しいテンソルを作成
        pooled_output = torch.zeros_like(tensor)

        # 有効なインデックスを取得
        valid_indices = subword_indices[subword_indices != self.SUBWORD_MASK_ID]
        if len(valid_indices) == 0:
            return pooled_output

        max_group_id = int(valid_indices.max().item())

        # バッチごとに処理
        for b in range(batch_size):
            # 各グループIDについて処理
            for group_id in range(max_group_id + 1):
                # グループに属する位置を特定
                group_mask = subword_indices[b] == group_id
                if not group_mask.any():
                    continue

                # アテンションマスクを適用
                valid_mask = group_mask & (attention_mask[b] == 1)
                if not valid_mask.any():
                    continue

                # グループ内の値を取得
                group_values = tensor[b][valid_mask]  # (num_tokens, vocab_size)

                if self.pooling_type == "max":
                    # グループ内で最大値を取る
                    pooled_values = torch.max(group_values, dim=0)[0]  # (vocab_size,)
                else:  # mean
                    # グループ内で平均を取る
                    pooled_values = torch.mean(group_values, dim=0)  # (vocab_size,)

                # プーリング結果をグループ内の全位置に設定
                # mask がある部分だけ更新
                pooled_output[b][group_mask] = pooled_values

        return pooled_output

    def forward(self, batch_inputs: dict, batch_size: int):
        subword_indices = create_subword_indices(
            batch_inputs["input_ids"],
            self.subword_token_ids,
        )

        logits = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

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

    current_subword_group_idx = -1
    for b in range(batch_size):
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
                current_subword_group_idx += 1
                in_subword_sequence = True
                # 直前のトークンも同じグループに
                if word_start_pos != -1:
                    subword_indices[b, word_start_pos : i + 1] = (
                        current_subword_group_idx
                    )

            # サブワードシーケンスの途中
            elif is_subword and in_subword_sequence:
                subword_indices[b, i] = current_subword_group_idx

            # サブワードシーケンスの終了
            if not is_subword and in_subword_sequence:
                word_start_pos = i
                in_subword_sequence = False

        # 最後の単語の処理
        if word_start_pos != -1 and not in_subword_sequence:
            subword_indices[b, word_start_pos] = -100

    return subword_indices
