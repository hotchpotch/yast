import torch

from yast.modeling.splade_subword import create_subword_indices


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
