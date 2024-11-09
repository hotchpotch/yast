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

    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments):
        """
        Args:
            pooling_type (str): プーリングの種類 ('max' または 'mean')
        """
        super().__init__(hf_model, model_args)
        self.relu = nn.ReLU()
        pooling_type = "max"
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

    def forward(self, output, attention_mask, subword_indices):
        """
        順伝播処理

        Args:
            output (torch.Tensor): 入力テンソル (batch_size, sequence_length, vocab_size)
            attention_mask (torch.Tensor): 注意マスク (batch_size, sequence_length)
            subword_indices (torch.Tensor): サブワードグループインデックス (batch_size, sequence_length)

        Returns:
            torch.Tensor: 出力テンソル (batch_size, vocab_size)
        """
        if output.dim() != 3 or attention_mask.dim() != 2:
            raise ValueError("Invalid input dimensions")

        if output.size(0) != attention_mask.size(0) or output.size(
            1
        ) != attention_mask.size(1):
            raise ValueError("Mismatched batch size or sequence length")

        # サブワードインデックスの値の検証
        if torch.any((subword_indices != self.SUBWORD_MASK_ID) & (subword_indices < 0)):
            raise ValueError(
                f"Invalid subword indices found. Must be either >= 0 or {self.SUBWORD_MASK_ID}"
            )

        # サブワード単位でのプーリング
        pooled = self.subword_pooling(output, subword_indices, attention_mask)

        # SPLADEの処理
        activated = torch.log(1 + self.relu(pooled))
        masked = activated * attention_mask.unsqueeze(-1)
        values, _ = torch.max(masked, dim=1)

        return values


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


# 使用例
def test_splade_subword():
    """
    サブワードプーリングのテスト関数
    """
    # サンプルのサブワードトークンID
    subword_token_ids = {1, 2, 3}  # 例: ID 1, 2, 3 がサブワードトークン

    # サンプルの入力データ
    token_ids = torch.tensor(
        [
            [10, 1, 2, 11, 3, 12],  # 例: [通常, サブ, サブ, 通常, サブ, 通常]
            [13, 14, 15, -100, -100, -100],  # 例: [通常, 通常, 通常, PAD, PAD, PAD]
        ]
    )

    # サブワードインデックスの生成
    subword_indices = create_subword_indices(token_ids, subword_token_ids)
    print("Generated subword indices:\n", subword_indices)

    # 残りのテストデータを生成
    batch_size, seq_len = token_ids.shape
    vocab_size = 32000
    # 勾配を有効にしたい
    output_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    attention_mask = (token_ids != -100).long()

    # モデルのテスト
    model_max = SpladeMaxPoolingWithSubword(pooling_type="max")
    model_mean = SpladeMaxPoolingWithSubword(pooling_type="mean")

    output_max = model_max(output_logits, attention_mask, subword_indices)
    output_mean = model_mean(output_logits, attention_mask, subword_indices)

    # check require_grad
    print("Max pooling requires_grad:", output_max.requires_grad)
    print("Mean pooling requires_grad:", output_mean.requires_grad)
    print("Max pooling output shape:", output_max.shape)
    print("Mean pooling output shape:", output_mean.shape)

    return output_max, output_mean, subword_indices


def test_gradient_flow():
    """
    勾配の流れをテストする関数 - 修正版
    """
    # サンプルデータの作成 - 明示的に勾配を有効化
    batch_size, seq_len, vocab_size = 2, 6, 100

    # ここが重要: requires_grad=True で初期化し、detach() を使って新しいリーフテンソルを作成
    output_logits = (
        torch.randn(batch_size, seq_len, vocab_size).requires_grad_(True).detach()
    )
    output_logits.requires_grad_(True)  # 明示的に勾配計算を有効化

    attention_mask = torch.ones(batch_size, seq_len)
    subword_indices = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2],  # バッチ1
            [0, 1, 2, -100, -100, -100],  # バッチ2
        ]
    )

    # デバイスの選択と移動
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpladeMaxPoolingWithSubword(pooling_type="max").to(device)
    output_logits = output_logits.to(device)
    attention_mask = attention_mask.to(device)
    subword_indices = subword_indices.to(device)

    # 勾配計算の前に、テンソルの状態を確認
    print("Before forward pass:")
    print("- requires_grad:", output_logits.requires_grad)
    print("- is_leaf:", output_logits.is_leaf)

    # 順伝播
    output = model(output_logits, attention_mask, subword_indices)

    print("\nAfter forward pass:")
    print("- Output requires_grad:", output.requires_grad)

    # 逆伝播のテスト
    loss = output.sum()
    loss.backward()

    # 勾配の確認 - リーフテンソルの勾配を確認
    grad_exists = output_logits.grad is not None
    if grad_exists:
        grad_nonzero = torch.any(output_logits.grad != 0).item()
        print("\nGradient check:")
        print("- Gradient exists:", grad_exists)
        print("- Gradient has non-zero values:", grad_nonzero)
        print("- Gradient magnitude:", output_logits.grad.abs().mean().item())
    else:
        print("\nNo gradient was computed!")

    # メモリ解放
    del output_logits, attention_mask, subword_indices, output, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    output_max, output_mean, subword_indices = test_splade_subword()
    # test_gradient_flow()
