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
        self.pooling_type = "max"  # or "mean"

    def create_group_index_tensor(
        self, subword_indices: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        グループインデックスとグループサイズの情報を作成
        バッチごとの連続した新しいグループIDを生成

        Args:
            subword_indices: サブワードグループのインデックス (batch_size, seq_len)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            tuple: (group_index, group_sizes, num_groups)
                group_index: 各位置がどのグループに属するかを示すテンソル
                group_sizes: 各グループのサイズ
                num_groups: 総グループ数
        """
        batch_size, seq_len = subword_indices.shape
        device = subword_indices.device

        # 有効なマスクを作成
        valid_mask = (subword_indices != self.SUBWORD_MASK_ID) & (attention_mask == 1)

        # バッチインデックスを作成
        batch_ids = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        )

        valid_batch_ids = batch_ids[valid_mask]
        valid_subword_ids = subword_indices[valid_mask]

        # バッチとサブワードの組み合わせで一意のグループIDを作成
        combined_ids = torch.stack([valid_batch_ids, valid_subword_ids], dim=1)
        unique_combined_ids, inverse_indices = torch.unique(
            combined_ids, dim=0, return_inverse=True
        )

        # グループサイズを計算
        group_sizes = torch.bincount(inverse_indices)
        num_groups = len(group_sizes)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Number of unique groups: {num_groups}")
            logger.debug(f"Group sizes: {group_sizes}")
            logger.debug(f"Max group id: {inverse_indices.max()}")

        return inverse_indices, group_sizes, num_groups

    def subword_pooling_optimized(
        self,
        tensor: torch.Tensor,
        subword_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        CUDAに最適化されたサブワードプーリング

        Args:
            tensor: 入力テンソル (batch_size, seq_len, vocab_size)
            subword_indices: サブワードグループのインデックス (batch_size, seq_len)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            torch.Tensor: プーリング適用後のテンソル (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, vocab_size = tensor.size()
        device = tensor.device
        dtype = tensor.dtype

        # 結果用のテンソルを初期化
        pooled_output = torch.zeros_like(tensor)

        # 有効なマスクを作成
        valid_mask = (subword_indices != self.SUBWORD_MASK_ID) & (attention_mask == 1)
        if not valid_mask.any():
            return pooled_output

        try:
            # グループ情報を取得
            group_ids, group_sizes, num_groups = self.create_group_index_tensor(
                subword_indices, attention_mask
            )

            # 有効な値を抽出
            valid_values = tensor[valid_mask]  # (num_valid, vocab_size)

            # プーリング処理
            if self.pooling_type == "max":
                pooled_values = torch.zeros(
                    (num_groups, vocab_size), device=device, dtype=dtype
                )
                scatter_indices = group_ids.unsqueeze(-1).expand(-1, vocab_size)

                pooled_values.scatter_reduce_(
                    0, scatter_indices, valid_values, reduce="amax"
                )
            else:  # mean
                pooled_sums = torch.zeros(
                    (num_groups, vocab_size), device=device, dtype=dtype
                )
                scatter_indices = group_ids.unsqueeze(-1).expand(-1, vocab_size)

                pooled_sums.scatter_reduce_(
                    0, scatter_indices, valid_values, reduce="sum"
                )
                pooled_values = pooled_sums / group_sizes.unsqueeze(-1).to(dtype)

            # 結果を元の形状に戻す
            pooled_output[valid_mask] = pooled_values[group_ids]

            return pooled_output

        except Exception as e:
            logger.error(f"Error in subword_pooling_optimized: {str(e)}")
            logger.error(
                f"tensor shape: {tensor.shape}, subword_indices shape: {subword_indices.shape}"
            )
            raise

    def forward(self, batch_inputs: dict, batch_size: int):
        """
        モデルのforward処理

        Args:
            batch_inputs: 入力バッチ
            batch_size: バッチサイズ

        Returns:
            モデルの出力
        """
        subword_indices = create_subword_indices(
            batch_inputs["input_ids"],
            self.subword_token_ids,
        )

        logits = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

        pooled = self.subword_pooling_optimized(logits, subword_indices, attention_mask)
        return self._forward_logits(pooled, attention_mask, batch_size)


def create_subword_indices(
    token_ids: torch.Tensor, subword_token_ids: set
) -> torch.Tensor:
    """
    トークンIDからサブワードインデックスを生成する
    サブワードを含む単語は同じインデックスでグループ化し、
    単独トークンは-100として扱う

    Args:
        token_ids (torch.Tensor): トークンID (batch_size, seq_len)
        subword_token_ids (set): サブワードとして扱うトークンIDのset

    Returns:
        torch.Tensor: サブワードインデックス (batch_size, seq_len)
            -100: 単独トークン（サブワードを含まない単語）やパディング
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
