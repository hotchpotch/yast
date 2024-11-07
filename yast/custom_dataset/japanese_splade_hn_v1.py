import logging
import random

from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForSpladeTraining

logger = logging.getLogger(__name__)

HADR_NEGATIVE_SCORE_DS = "hotchpotch/japanese-splade-v1-hard-negatives"
DS_SPIT = "train"
# QUERY_DS = "mmarco-dataset"
# COLLECTION_DS = "mmarco-collection"

NEG_SCORE_TH = 0.3
POS_SCORE_TH = 0.7
NEG_FILTER_COUNT = 7


def _map_filter_score(example, neg_score_th: float, pos_score_th: float):
    # neg_score: list[float] = example["neg.score"]
    # neg_score_filtered_index = [
    #     i for i, score in enumerate(neg_score) if score < neg_score_th
    # ]
    # # same pos_score
    # pos_score = example["pos.score"]
    # pos_score_filtered_index = [
    #     i for i, score in enumerate(pos_score) if score > pos_score_th
    # ]
    neg_score_top50 = example[
        "score.bge-reranker-v2-m3.neg_ids.japanese-splade-base-v1-mmarco-only.top50"
    ]
    neg_score_other50 = example[
        "score.bge-reranker-v2-m3.neg_ids.japanese-splade-base-v1-mmarco-only.other50"
    ]
    pos_score = example["score.bge-reranker-v2-m3.pos_ids"]
    neg_score_top50_filtered_index = [
        i for i, score in enumerate(neg_score_top50) if score < neg_score_th
    ]
    neg_score_other50_filtered_index = [
        i for i, score in enumerate(neg_score_other50) if score < neg_score_th
    ]
    pos_score_filtered_index = [
        i for i, score in enumerate(pos_score) if score > pos_score_th
    ]

    data = {
        **example,
        # "neg.score": [neg_score[i] for i in neg_score_filtered_index],
        # "neg": [example["neg"][i] for i in neg_score_filtered_index],
        # "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        # "pos": [example["pos"][i] for i in pos_score_filtered_index],
        "neg.score.top50": [neg_score_top50[i] for i in neg_score_top50_filtered_index],
        "neg.top50": [
            example["neg_ids.japanese-splade-base-v1-mmarco-only.top50"][i]
            for i in neg_score_top50_filtered_index
        ],
        "neg.score.other50": [
            neg_score_other50[i] for i in neg_score_other50_filtered_index
        ],
        "neg.other50": [
            example["neg_ids.japanese-splade-base-v1-mmarco-only.other50"][i]
            for i in neg_score_other50_filtered_index
        ],
        "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        "pos": [example["pos_ids"][i] for i in pos_score_filtered_index],
    }
    # XXX:
    # posは、neg_score_top50 から > 0.95 のものでサンプリングしても良いのでは?

    if len(pos_score_filtered_index) == 0:
        # neg_score_top50 の最大値と、その index を取得
        max_score = max(neg_score_top50)
        max_score_index = neg_score_top50.index(max_score)
        if max_score >= 0.9:
            # pos が閾値以上のものがなく、かつ十分なスコアが neg にある場合は、それを pos とする
            data["pos"] = [
                example["neg_ids.japanese-splade-base-v1-mmarco-only.top50"][
                    max_score_index
                ]
            ]
            data["pos.score"] = [max_score]

    return data


def _filter_score(example, net_filter_count: int):
    # neg のカウントがN以上で、pos のカウントが1以上のものを返す
    return (
        len(example["neg.other50"] + example["neg.top50"]) >= net_filter_count
        and len(example["pos"]) >= 1
    )


class JapaneseSpladeHardNegativesV1(DatasetForSpladeTraining):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):

        self.query_max_len = args.dataset_options.get("query_max_len", 256)
        self.doc_max_len = args.dataset_options.get("doc_max_len", 1024)
        train_data = args.train_data

        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)
        dataset_name = dataset_options.get("dataset_name", "mmarco")
        logger.info(f"Initializing {dataset_name} hard_negative dataset")

        query_ds_name = f"{dataset_name}-dataset"
        collection_ds_name = f"{dataset_name}-collection"

        # lang = train_data["lang"]
        # reranker_name = train_data["reranker"]
        neg_score_th = train_data.get("neg_score_th", NEG_SCORE_TH)
        pos_score_th = train_data.get("pos_score_th", POS_SCORE_TH)
        net_filter_count = train_data.get("net_filter_count", NEG_FILTER_COUNT)

        ds = load_dataset(HADR_NEGATIVE_SCORE_DS, query_ds_name, split=DS_SPIT)
        self.collection_ds = load_dataset(
            HADR_NEGATIVE_SCORE_DS, collection_ds_name, split=DS_SPIT
        )
        ds = ds.map(
            _map_filter_score,
            num_proc=11,  # type: ignore
            fn_kwargs={"neg_score_th": neg_score_th, "pos_score_th": pos_score_th},
        )  # type: ignore
        ds = ds.filter(
            _filter_score, num_proc=11, fn_kwargs={"net_filter_count": net_filter_count}
        )  # type: ignore
        logger.info(f"Filtered dataset size: {len(ds)}")

        super().__init__(args, tokenizer, ds)  # type: ignore

    def get_collection_text(self, doc_id: int) -> str:
        text = self.collection_ds[doc_id]["text"]  # type: ignore
        return text  # type: ignore

    def __getitem__(self, item) -> list[dict]:
        query = self.dataset[item]["anc"]
        pos_ids = self.dataset[item]["pos"]
        pos_ids_score = self.dataset[item]["pos.score"]
        # neg_ids = self.dataset[item]["neg"]
        # neg_ids_score = self.dataset[item]["neg.score"]
        neg_ids_top50 = self.dataset[item]["neg.top50"]
        neg_ids_score_top50 = self.dataset[item]["neg.score.top50"]

        # N をneg_ids_top50からrandom sampling
        top_50_count = 4
        if len(neg_ids_top50) < top_50_count:
            top_50_count = len(neg_ids_top50)
        neg_ids_top50_sampled = random.sample(neg_ids_top50, top_50_count)
        neg_ids_score_top50_sampled = [
            neg_ids_score_top50[neg_ids_top50.index(id_)]
            for id_ in neg_ids_top50_sampled
        ]

        # 7 じゃなくて、train_group_size - 1 にする
        other_50_count = 7 - top_50_count
        neg_ids_other50 = self.dataset[item]["neg.other50"]
        neg_ids_score_other50 = self.dataset[item]["neg.score.other50"]
        if len(neg_ids_other50) < other_50_count:
            other_50_count = len(neg_ids_other50)
        neg_ids_other50_sampled = random.sample(neg_ids_other50, other_50_count)
        neg_ids_score_other50_sampled = [
            neg_ids_score_other50[neg_ids_other50.index(id_)]
            for id_ in neg_ids_other50_sampled
        ]

        neg_ids = neg_ids_top50_sampled + neg_ids_other50_sampled
        neg_ids_score = neg_ids_score_top50_sampled + neg_ids_score_other50_sampled

        pos_texts = [self.get_collection_text(pos_id) for pos_id in pos_ids]
        neg_texts = [self.get_collection_text(neg_id) for neg_id in neg_ids]

        if self.binarize_label:
            pos_ids_score = [1.0] * len(pos_ids_score)
            neg_ids_score = [0.0] * len(neg_ids_score)

        return self.create_batch_inputs(
            query,
            pos_texts,
            neg_texts,
            pos_ids_score,
            neg_ids_score,
        )