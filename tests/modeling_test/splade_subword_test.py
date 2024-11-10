import pytest
import torch
from transformers import AutoTokenizer

from yast.arguments import ModelArguments
from yast.modeling.splade_subword import SpladeSubword, create_subword_indices


def test_create_subword_indices():
    # Sample subword token IDs
    subword_token_ids = {1, 2, 3}  # Example: IDs 1, 2, 3 are subword tokens

    # Sample input data
    token_ids = torch.tensor(
        [
            [
                9,
                1,
                2,
                10,
                11,
                3,
                12,
            ],  # Example: [normal, subword, subword, normal, normal, subword, normal]
            [
                13,
                14,
                1,
                -100,
                -100,
                -100,
                -100,
            ],  # Example: [normal, subword, normal, PAD, PAD, PAD, PAD]
        ]
    )

    subword_indices = create_subword_indices(token_ids, subword_token_ids)

    expected_indices = torch.tensor(
        [
            [0, 0, 0, -100, 1, 1, -100],
            [
                -100,
                0,
                0,
                -100,
                -100,
                -100,
                -100,
            ],
        ]
    )

    assert torch.equal(subword_indices, expected_indices)


@pytest.mark.only
def test_splade_subword():
    hf_model_name = "hotchpotch/japanese-splade-base-v1"
    model_args = ModelArguments(model_name_or_path=hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        use_fast=False,
    )

    model = SpladeSubword.from_pretrained(
        model_args,
        model_args.model_name_or_path,
    )
    model.tokenizer = tokenizer

    # model.forward 用のデータを作成
    batch_inputs = tokenizer(
        ["こんにちは、世界!", "こんばんは、日本と世界!"],
        padding=True,
        return_tensors="pt",
    )

    queries, docs = model.forward(
        batch_inputs,  # type: ignore
        batch_size=2,
    )
    assert queries.shape == torch.Size([2, 1, 32768])
    assert docs.shape == torch.Size([2, 0, 32768])
