import logging

import torch
from torch import nn
from transformers import PreTrainedModel

from ..arguments import ModelArguments
from .splade import Splade

logger = logging.getLogger(__name__)


class SpladeSubword(Splade):
    SUBWORD_MASK_ID = -100

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        super().__init__(hf_model, model_args)
        self.relu = nn.ReLU()
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

        # subword_indices = batch_inputs["subword_indices"]
        subword_indices = batch_inputs.pop("subword_indices")

        logits = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

        pooled = self.subword_pooling_optimized(logits, subword_indices, attention_mask)
        return self._forward_logits(pooled, attention_mask, batch_size)
