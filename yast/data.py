"""
Code adapted from: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/data.py
License: MIT License
"""

import functools
import importlib
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Type, cast

import torch
import torch.utils.data
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .arguments import DataArguments

logger = logging.getLogger(__name__)


POSITIVE_KEYS = ["pos", "positive", "positives"]
NEGATIVE_KEYS = ["neg", "negative", "negatives"]
QUERY_KEYS = ["query", "qry", "question", "q"]
POSITIVE_SCORE_KEYS = [
    "pos_score",
    "positive_score",
    "positive_scores",
    "positives_score",
]
NEGATIVE_SCORE_KEYS = [
    "neg_score",
    "negative_score",
    "negative_scores",
    "negatives_score",
]


class DatasetForSpladeTraining(torch.utils.data.Dataset):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Dataset | None = None,
    ):
        if not dataset:
            train_data = args.train_data  # list or str
            if isinstance(train_data, list):
                datasets = []
                for target in train_data:
                    logger.info(f"Loading {target}")
                    datasets.append(self.load_dataset(target))
                self.dataset = concatenate_datasets(datasets)
            else:
                logger.info(f"Loading {train_data}")
                self.dataset = self.load_dataset(train_data)
        else:
            self.dataset = dataset

        self.dataset: Dataset = cast(Dataset, self.dataset)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        self.subword_token_ids = set()

        if args.create_subword_indices:
            subword_prefix = "##"  # bert subword prefix
            for token in tokenizer.get_vocab():
                if token.startswith(subword_prefix):
                    self.subword_token_ids.add(tokenizer.convert_tokens_to_ids(token))
            logger.info(f"Subword token count: {len(self.subword_token_ids)}")

    def load_dataset(self, target_name: str) -> Dataset:
        if target_name.endswith(".jsonl") or target_name.endswith(".json"):
            logger.info(f"Loading JSON dataset from {target_name}")
            return load_dataset("json", data_files=target_name)["train"]  # type: ignore
        elif os.path.isdir(target_name):
            datasets = []
            target_files = os.listdir(target_name)
            if any([f.endswith(".arrow") for f in target_files]):
                # has arrow files
                logger.info(f"Loading dataset from directory {target_name}")
                target_ds = load_from_disk(target_name)
                logger.info(f"Loaded {target_name}: {len(target_ds)} examples")
                datasets.append(target_ds)
            else:
                for target in target_files:
                    full_path = os.path.join(target_name, target)
                    logger.info(f"Loading {full_path}")
                    target_ds = self.load_dataset(full_path)
                    logger.info(f"Loaded {full_path}: {len(target_ds)} examples")
                    datasets.append(target_ds)
            return concatenate_datasets(datasets)
        else:
            logger.info(f"Loading dataset {target_name} with split='train'")
            return load_dataset(target_name, split="train")  # type: ignore

    @property
    @functools.lru_cache(maxsize=None)
    def query_key(self):
        for key in QUERY_KEYS:
            if key in self.dataset.column_names:
                return key
        raise ValueError("Query key not found")

    @property
    @functools.lru_cache(maxsize=None)
    def positive_key(self):
        for key in POSITIVE_KEYS:
            if key in self.dataset.column_names:
                return key
        raise ValueError("Positive key not found")

    @property
    @functools.lru_cache(maxsize=None)
    def negative_key(self):
        for key in NEGATIVE_KEYS:
            if key in self.dataset.column_names:
                return key
        raise ValueError("Negative key not found")

    @property
    @functools.lru_cache(maxsize=None)
    def positive_score_key(self):
        for key in POSITIVE_SCORE_KEYS:
            if key in self.dataset.column_names:
                return key
        return None

    @property
    @functools.lru_cache(maxsize=None)
    def negative_score_key(self):
        for key in NEGATIVE_SCORE_KEYS:
            if key in self.dataset.column_names:
                return key
        return None

    def create_one_example(
        self, encoding: str, max_length: int | None = None
    ) -> BatchEncoding:
        if not max_length:
            max_length = self.args.max_length
        item = self.tokenizer.encode_plus(
            encoding,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        if len(self.subword_token_ids) > 0:
            item["subword_indices"] = create_subword_indices(
                torch.tensor(item["input_ids"]).unsqueeze(0), self.subword_token_ids
            ).squeeze(0)

        return item

    def create_batch_inputs(
        self,
        query: str,
        pos_texts: list[str],
        neg_texts: list[str],
        pos_ids_score: list[float],
        neg_ids_score: list[float],
    ):
        pos_size = min(self.args.train_max_positive_size, len(pos_texts))
        neg_size = self.args.train_group_size - pos_size

        pos_targets = list(zip(pos_texts, pos_ids_score))
        if len(pos_targets) >= pos_size:
            pos_positions = random.sample(range(len(pos_targets)), pos_size)
        else:
            pos_positions = random.choices(range(len(pos_targets)), k=pos_size)
        pos_texts = [pos_targets[i][0] for i in pos_positions]
        pos_ids_score = [pos_targets[i][1] for i in pos_positions]

        neg_targets = list(zip(neg_texts, neg_ids_score))
        if len(neg_targets) >= neg_size:
            neg_positions = random.sample(range(len(neg_targets)), neg_size)
        else:
            neg_positions = random.choices(range(len(neg_targets)), k=neg_size)

        neg_texts = [neg_targets[i][0] for i in neg_positions]
        neg_ids_score = [neg_targets[i][1] for i in neg_positions]

        labels = [float("nan")] + pos_ids_score + neg_ids_score

        batch_inputs = []
        batch_inputs.append(self.create_one_example(query, self.args.max_query_length))
        for text in pos_texts + neg_texts:
            batch_inputs.append(self.create_one_example(text, self.args.max_length))
        for label, batch_input in zip(labels, batch_inputs):
            batch_input["label"] = label
        return batch_inputs

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[dict]:
        query = self.dataset[item][self.query_key]
        pos_texts = self.dataset[item][self.positive_key]
        if not isinstance(pos_texts, list):
            pos_texts = [pos_texts]
        pos_scores = self.dataset[item].get(self.positive_score_key, [])
        if not isinstance(pos_scores, list):
            pos_scores = [pos_scores]
        neg_texts = self.dataset[item][self.negative_key]
        if not isinstance(neg_texts, list):
            neg_texts = [neg_texts]
        neg_scores = self.dataset[item].get(self.negative_score_key, [])
        if not isinstance(neg_scores, list):
            neg_scores = [neg_scores]

        return self.create_batch_inputs(
            query, pos_texts, neg_texts, pos_scores, neg_scores
        )


@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(self, features):
        if isinstance(features[0], list):
            features = sum(features, [])  # type: ignore

        # subword_indices がある場合、
        # サブワードインデックスのパディング処理を追加
        if "subword_indices" in features[0]:
            max_length = max(len(f["input_ids"]) for f in features)
            for feature in features:
                padding_length = max_length - len(feature["input_ids"])
                if padding_length > 0:
                    feature["subword_indices"] = (
                        feature["subword_indices"].tolist() + [-100] * padding_length
                    )

        return super().__call__(features)


def detect_dataset_klass(dataset_path: str) -> Type[DatasetForSpladeTraining]:
    module_path, class_name = dataset_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    dataset_class = getattr(module, class_name)
    return dataset_class


def create_dateset_from_args(
    args: DataArguments, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
) -> DatasetForSpladeTraining:
    train_data = args.train_data
    if isinstance(train_data, str):
        target_ds = DatasetForSpladeTraining(args, tokenizer)
    elif isinstance(train_data, list):
        target_ds_list = []
        for target_train_data in train_data:
            dataset_class_args = deepcopy(args)
            if isinstance(target_train_data, str):
                dataset_class_args.train_data = target_train_data
                target_ds_list.append(
                    DatasetForSpladeTraining(dataset_class_args, tokenizer)
                )
            elif isinstance(target_train_data, dict):
                dataset_class_name = target_train_data.get("dataset_class")
                dataset_options = target_train_data.get("dataset_options", {})
                # merge dataset_options
                dataset_class_args.dataset_options.update(dataset_options)

                if not dataset_class_name:
                    raise ValueError(f"dataset_class is required, {target_train_data}")
                dataset_class_train_data = target_train_data.get("train_data")
                if dataset_class_train_data:
                    dataset_class_args.train_data = dataset_class_train_data
                dataset_klass = detect_dataset_klass(dataset_class_name)
                target_ds_list.append(dataset_klass(dataset_class_args, tokenizer))
            else:
                raise ValueError(f"Invalid type {target_train_data}")
        target_ds = torch.utils.data.ConcatDataset(target_ds_list)
    return target_ds  # type: ignore


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
