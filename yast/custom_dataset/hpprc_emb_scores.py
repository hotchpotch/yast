from functools import cache
from typing import cast

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForSpladeTraining

HPPRC_EMB_DS = "hpprc/emb"
SCORE_DS = "hotchpotch/hpprc_emb-scores"


@cache
def get_dataset_subsets(dataset_name):
    api = HfApi()
    dataset_info = api.dataset_info(dataset_name)

    if "configs" in dataset_info.card_data:  # type: ignore
        return [config["config_name"] for config in dataset_info.card_data["configs"]]  # type: ignore
    elif "dataset_info" in dataset_info.card_data:  # type: ignore
        return [info["config_name"] for info in dataset_info.card_data["dataset_info"]]  # type: ignore
    else:
        return []


def get_datasets(target_name: str):
    subsets = get_dataset_subsets(SCORE_DS)
    target_subsets = [subset for subset in subsets if subset.startswith(target_name)]
    if not target_subsets:
        raise ValueError(f"Subset not found: {target_name}")
    target_subset = target_subsets[0]
    target_base_name, revision = target_subset.rsplit("-dataset__", 1)
    score_ds = load_dataset(
        SCORE_DS,
        name=target_subset,
        split="train",
    )
    if target_name.startswith("quiz-") or target_name.startswith("mkqa"):
        collection_name = "qa-collection"
    else:
        collection_name = f"{target_base_name}-collection"
    hpprc_emb_ds = load_dataset(
        HPPRC_EMB_DS,
        name=collection_name,
        split="train",
        revision=revision,
    )
    return hpprc_emb_ds, score_ds


TARGET_POS_ID_KEYS = [
    "pos_ids",
    "pos_ids.original",
    "pos_ids.me5-large",
    "pos_ids.bm25",
]
TARGET_NEG_ID_KEYS = [
    "neg_ids",
    "neg_ids.original",
    "neg_ids.me5-large",
    "neg_ids.bm25",
]


def map_data(
    example,
    target_score_keys: list[str] = ["ruri-reranker-large", "bge-reranker-v2-m3"],
    pos_id_score_threshold: float = 0.7,
    neg_id_score_threshold: float = 0.3,
):
    target_id_keys = TARGET_POS_ID_KEYS + TARGET_NEG_ID_KEYS
    # 対象の target_score の平均値を取得
    target_score_dict = {}
    for id_key in target_id_keys:
        scores_key_values = []
        for score_key in target_score_keys:
            full_score_key = f"score.{score_key}.{id_key}"
            scores = example.get(full_score_key, [])
            if len(scores) > 0:
                scores_key_values.append(np.array(scores))
        if len(scores_key_values) > 0:
            if len(scores_key_values) != len(target_score_keys):
                raise ValueError(
                    f"len(scores_key_values) != len(target_score_keys): {len(scores_key_values)} != {len(target_score_keys)}"
                )
            mean_scores = np.array(scores_key_values).T.mean(axis=1)
            target_score_dict[id_key] = mean_scores.tolist()

    filtered_target_ids_dict = {}
    # 閾値でフィルタリング
    for id_key in target_score_dict.keys():
        target_score_ids = example[id_key]
        target_scores = target_score_dict[id_key]
        if "pos_ids" in id_key:
            filtered_target_scores_indexes = [
                i
                for i, score in enumerate(target_scores)
                if score >= pos_id_score_threshold
            ]
        else:
            filtered_target_scores_indexes = [
                i
                for i, score in enumerate(target_scores)
                if score <= neg_id_score_threshold
            ]
        filtered_target_ids = [
            target_score_ids[i] for i in filtered_target_scores_indexes
        ]
        filtered_target_ids_dict[id_key] = filtered_target_ids
        target_score_dict[id_key] = [
            target_scores[i] for i in filtered_target_scores_indexes
        ]
    result_pos_ids = []
    result_pos_ids_score = []
    result_neg_ids = []
    result_neg_ids_score = []
    for id_key in target_score_dict.keys():
        # pos_ids, neg_ids ともに重複IDがあるので、その場合は追加しない
        if "pos_ids" in id_key and id_key not in result_pos_ids:
            result_pos_ids += filtered_target_ids_dict[id_key]
            result_pos_ids_score += target_score_dict[id_key]
        elif "neg_ids" in id_key and id_key not in result_neg_ids and id_key:
            result_neg_ids += filtered_target_ids_dict[id_key]
            result_neg_ids_score += target_score_dict[id_key]
    # print("pos_ids", result_pos_ids)
    # print("neg_ids", result_neg_ids)
    # print("result_pod_ids", result_pos_ids_score)
    # print("result_neg_ids", result_neg_ids_score)
    return {
        "anc": example["anc"],
        "pos_ids": result_pos_ids,
        "pos_ids.score": result_pos_ids_score,
        "neg_ids": result_neg_ids,
        "neg_ids.score": result_neg_ids_score,
    }


def filter_data(example):
    neg_ids = example["neg_ids"]
    if len(neg_ids) < 8:
        return False
    pos_ids = example["pos_ids"]
    if len(pos_ids) == 0:
        return False
    return True


class HpprcEmbScoresDataset(DatasetForSpladeTraining):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        seed: int = 42,
    ):
        train_data = args.train_data
        # train data は list
        if not isinstance(train_data, list):
            raise ValueError("train_data must be a list")
        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)
        all_ds = []
        target_emb_ds = {}
        for target in train_data:
            if not isinstance(target, dict):
                raise ValueError("train_data must be a list of dictionaries")
            subset = target["subset"]
            print("Processing...", subset)
            n = target.get("n", None)
            emb_ds, score_ds = get_datasets(subset)
            score_ds = cast(Dataset, score_ds)
            score_ds = score_ds.map(
                map_data, num_proc=11, remove_columns=score_ds.column_names
            )  # type: ignore
            target_emb_ds[subset] = emb_ds
            aug_factor = target.get("aug_factor", 1.0)
            if aug_factor != 1.0:
                if n is not None:
                    print(
                        "Warning: aug_factor is ignored because n is specified, skip aug_factor args",
                        subset,
                    )
                else:
                    n = int(len(score_ds) * aug_factor)
                    print(f"Augment dataset: {subset} @{aug_factor}")
            if n is not None:
                if n > len(score_ds):
                    print(f"Expand dataset: {subset} orig: {len(score_ds)} => {n}")
                    score_ds_expand = []
                    c = n // len(score_ds)
                    r = n % len(score_ds)
                    for _ in range(c):
                        score_ds_expand.append(score_ds.shuffle(seed=seed))
                    score_ds_expand.append(score_ds.shuffle(seed=seed).select(range(r)))
                    score_ds = concatenate_datasets(score_ds_expand)
                    assert len(score_ds) == n
                else:
                    score_ds = score_ds.shuffle(seed=seed).select(range(n))  # type: ignore
            before_filter_len = len(score_ds)
            score_ds = score_ds.filter(filter_data, num_proc=11)
            print(
                "Filtered",
                subset,
                before_filter_len,
                len(score_ds),
                f"{len(score_ds) / before_filter_len:.2f}",
            )
            subsets = [subset] * len(score_ds)
            score_ds = score_ds.add_column("subset", subsets)  # type: ignore
            all_ds.append(score_ds)
            print("Loaded", subset, len(score_ds))
        self.target_emb_ds = target_emb_ds
        ds = concatenate_datasets(all_ds)
        super().__init__(args, tokenizer, ds)

    def get_text_by_subset(self, subset: str, idx: int, max_len: int = 1024) -> str:
        emb_ds = self.target_emb_ds[subset]
        row = emb_ds[idx]
        text = row["text"]
        title = row.get("title", None)
        if title:
            text = title + " " + text
        return text[0:max_len]

    def __getitem__(self, item) -> list[dict]:
        subset = self.dataset[item]["subset"]
        query = self.dataset[item]["anc"]
        pos_ids = self.dataset[item]["pos_ids"]
        pos_ids_score = self.dataset[item]["pos_ids.score"]
        neg_ids = self.dataset[item]["neg_ids"]
        neg_ids_score = self.dataset[item]["neg_ids.score"]
        pos_texts = [self.get_text_by_subset(subset, pos_id) for pos_id in pos_ids]
        neg_texts = [self.get_text_by_subset(subset, neg_id) for neg_id in neg_ids]

        if self.binarize_label:
            pos_ids_score = [1.0] * len(pos_ids_score)
            neg_ids_score = [0.0] * len(neg_ids_score)

        return self.create_butch_inputs(
            query,
            pos_texts,
            neg_texts,
            pos_ids_score,
            neg_ids_score,
        )
