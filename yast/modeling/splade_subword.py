import logging

import torch
from torch import nn
from transformers import PreTrainedModel

from ..arguments import ModelArguments
from .splade import Splade

logger = logging.getLogger(__name__)


class SpladeSubword(Splade):
    POOLING_TYPES = ["max", "mean"]  # 使用可能なプーリングタイプ
    SUBWORD_MASK_ID = -100  # 無視するサブワードのマスクID

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        super().__init__(hf_model, model_args)
        subword_pooling = model_args.subword_pooling
        if subword_pooling is not None and subword_pooling not in self.POOLING_TYPES:
            raise ValueError(
                f"無効なプーリングタイプ: {subword_pooling}. {self.POOLING_TYPES} の中から選択してください。"
            )
        self.pooling_type = subword_pooling

    def _aggregate_subwords(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        subword_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorchのscatter操作を使用してサブワードのlogitsを最適化して集約します。

        Args:
            logits (torch.Tensor): splade_max後の [batch_size, vocab_size] 形状のlogitsテンソル。
            input_ids (torch.Tensor): [batch_size, sequence_length] 形状の入力トークンID。
            subword_indices (torch.Tensor): [batch_size, sequence_length] 形状のサブワードグループインデックス。

        Returns:
            torch.Tensor: 集約後のlogitsテンソル。
        """
        batch_size, vocab_size = logits.size()
        device = logits.device
        dtype = logits.dtype  # 一貫したデータ型を確保

        # 有効なサブワード位置のマスクを作成
        mask = subword_indices != self.SUBWORD_MASK_ID  # [B, S]

        # 有効な位置のインデックスを取得
        valid_positions = mask.nonzero(
            as_tuple=False
        )  # [N, 2] 各行は (batch_idx, seq_idx)
        if valid_positions.numel() == 0:
            return logits  # 集約すべきサブワードがない場合は元のlogitsを返す

        batch_indices = valid_positions[:, 0]  # [N]
        seq_indices = valid_positions[:, 1]  # [N]

        # 該当するグループインデックスとトークンIDを抽出
        group_indices = subword_indices[batch_indices, seq_indices]  # [N]
        token_ids = input_ids[batch_indices, seq_indices]  # [N]

        # 選択されたトークンIDに対応するlogitsを取得
        selected_logits = logits[batch_indices, token_ids]  # [N]

        # 各バッチごとにグループIDをオフセットしてユニークなグループ識別子を計算
        max_group_tensor = group_indices.max()
        max_group = (
            max_group_tensor.item()
            if torch.is_tensor(max_group_tensor)
            else max_group_tensor
        )
        if max_group < 0:
            max_group = 0  # マスクされていないグループがない場合をハンドリング

        # グループ識別子を一意にするためにバッチオフセットを加算
        group_offset = batch_indices * (max_group + 1)  # [N]
        unique_group_ids = group_offset + group_indices  # [N]

        # ユニークなグループの数を計算
        num_unique_groups = batch_size * (max_group + 1)  # .item()を削除

        if self.pooling_type == "max":
            # -infで初期化し、logitsと同じdtypeを設定
            pooled_values = torch.full(
                (num_unique_groups,), -float("inf"), device=device, dtype=dtype
            )
            # scatter_reduceを使用して各グループの最大値を計算
            pooled_values = pooled_values.scatter_reduce(
                dim=0,
                index=unique_group_ids,
                src=selected_logits,
                reduce="amax",
                include_self=True,
            )
        elif self.pooling_type == "mean":
            # 合計とカウントのテンソルを初期化し、logitsと同じdtypeを設定
            sum_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)
            count_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)
            # 各グループの合計を計算
            sum_pooled = sum_pooled.scatter_add(0, unique_group_ids, selected_logits)
            # 各グループのカウントを計算
            count_pooled = count_pooled.scatter_add(
                0, unique_group_ids, torch.ones_like(selected_logits)
            )
            # 平均値を計算（ゼロ除算を防ぐ）
            pooled_values = sum_pooled / torch.clamp(count_pooled, min=1)
        else:
            raise ValueError(
                f"サポートされていないプーリングタイプ: {self.pooling_type}"
            )

        # プールされた値を各トークンにマッピング
        pooled_values_per_token = pooled_values[unique_group_ids]  # [N]

        # バッチ内の複数のトークンを処理するためにグローバルなトークンIDを計算
        global_token_ids = batch_indices * vocab_size + token_ids  # [N]

        # 各トークンごとの最大プール値を保持するテンソルを初期化（-infで初期化）
        max_pooled_per_token = torch.full(
            (batch_size * vocab_size,), -float("inf"), device=device, dtype=dtype
        )
        # scatter_reduceを使用して各トークンの最大プール値を計算
        max_pooled_per_token = max_pooled_per_token.scatter_reduce(
            dim=0,
            index=global_token_ids,
            src=pooled_values_per_token,
            reduce="amax",
            include_self=True,
        )

        # テンソルを [batch_size, vocab_size] にリシェイプ
        max_pooled_per_token = max_pooled_per_token.view(batch_size, vocab_size)

        # 元のlogitsとプールされた値の最大値を取って更新
        new_logits = torch.maximum(logits, max_pooled_per_token)

        return new_logits

    def forward(self, batch_inputs: dict, batch_size: int):
        """
        モデルのフォワード処理

        Args:
            batch_inputs: 'input_ids', 'attention_mask', 'subword_indices' を含む入力バッチ。
            batch_size: バッチサイズ。

        Returns:
            クエリとドキュメントの表現のタプル。
        """
        # 'subword_indices' をバッチ入力から取り出す
        subword_indices = batch_inputs.pop("subword_indices")
        input_ids = batch_inputs["input_ids"]

        # Hugging Faceモデルを使用して出力を取得
        output = self.hf_model(**batch_inputs, return_dict=True).logits
        attention_mask = batch_inputs["attention_mask"]

        # splade_max 関数を適用
        logits = self.splade_max(output, attention_mask)

        # サブワードの集約処理を実行
        if self.pooling_type is not None:
            logits = self._aggregate_subwords(logits, input_ids, subword_indices)

        # クエリとドキュメントの表現を取得
        query, docs = self._logit_to_query_docs(logits, batch_size)
        return query, docs
