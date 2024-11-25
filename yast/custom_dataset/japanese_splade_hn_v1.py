import logging
import random

from datasets import concatenate_datasets, load_dataset
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
NEG_POS_SCORE_TH = 0.95

TOP_100_SAMPLING_COUNT = 4  # top100 から hard negative としてサンプリングする数


def _map_score_with_hard_positives(
    example,
    neg_score_th: float,
    pos_score_th: float,
    neg_pos_score_th: float,
    hard_positives: bool,
    target_model_name: str = "japanese-splade-base-v1-mmarco-only",
):
    neg_score_top100 = example[
        f"score.bge-reranker-v2-m3.neg_ids.{target_model_name}.top100"
    ]
    neg_score_other100 = example[
        f"score.bge-reranker-v2-m3.neg_ids.{target_model_name}.other100"
    ]
    pos_score = example["score.bge-reranker-v2-m3.pos_ids"]
    neg_score_top100_filtered_index = [
        i for i, score in enumerate(neg_score_top100) if score < neg_score_th
    ]
    neg_score_other100_filtered_index = [
        i for i, score in enumerate(neg_score_other100) if score < neg_score_th
    ]
    pos_score_filtered_index = [
        i for i, score in enumerate(pos_score) if score > pos_score_th
    ]

    # hard positives はまずは、neg.other100 から取得する
    hard_positives_ids = example[f"neg_ids.{target_model_name}.other100"]
    hard_positives_scores = neg_score_other100
    hard_positives_score_filtered_index = [
        i for i, score in enumerate(hard_positives_scores) if score > neg_pos_score_th
    ]

    # top100 では hard_positives としては弱いので、使わない
    # if len(hard_positives_score_filtered_index) == 0:
    #     # neg.other100 に hard positives に該当するスコアがない場合、top100 から取得する
    #     hard_positives_ids = example[
    #         f"neg_ids.{target_model_name}.top100"
    #     ]
    #     hard_positives_scores = neg_score_top100
    #     hard_positives_score_filtered_index = [
    #         i
    #         for i, score in enumerate(hard_positives_scores)
    #         if score > neg_pos_score_th
    #     ]

    data = {
        **example,
        "neg.score.top100": [
            neg_score_top100[i] for i in neg_score_top100_filtered_index
        ],
        "neg.top100": [
            example[f"neg_ids.{target_model_name}.top100"][i]
            for i in neg_score_top100_filtered_index
        ],
        "neg.score.other100": [
            neg_score_other100[i] for i in neg_score_other100_filtered_index
        ],
        "neg.other100": [
            example[f"neg_ids.{target_model_name}.other100"][i]
            for i in neg_score_other100_filtered_index
        ],
        "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        "pos": [example["pos_ids"][i] for i in pos_score_filtered_index],
    }
    if hard_positives and len(hard_positives_score_filtered_index) > 0:
        # hard_positives flag がある
        # かつ hard_positives としてふさわしいスコアがある場合、pos.score, pos を neg に置き換える
        data["pos.score"] = [
            hard_positives_scores[i] for i in hard_positives_score_filtered_index
        ]
        data["pos"] = [
            hard_positives_ids[i] for i in hard_positives_score_filtered_index
        ]
    elif len(pos_score_filtered_index) == 0:
        # neg_score_top100 の最大値と、その index を取得
        max_score = max(neg_score_top100)
        max_score_index = neg_score_top100.index(max_score)
        if max_score >= neg_pos_score_th:
            # pos が閾値以上のものがなく、かつ十分なスコアが neg にある場合は、それを pos とする
            data["pos"] = [
                example[f"neg_ids.{target_model_name}.top100"][max_score_index]
            ]
            data["pos.score"] = [max_score]
        elif len(hard_positives_score_filtered_index) > 0:
            # neg_score_top100 にも hard_positives にも該当するスコアがない場合、
            # hard_positives_score_filtered_index から pos を1つランダムに追加する
            hard_positve_index = random.choice(hard_positives_score_filtered_index)
            max_score = hard_positives_scores[hard_positve_index]
            data["pos.score"] = [max_score]
            data["pos"] = [hard_positives_ids[hard_positve_index]]
    return data


def _filter_score(example, net_filter_count: int):
    # neg のカウントがN以上で、pos のカウントが1以上のものを返す
    return (
        len(example["neg.other100"] + example["neg.top100"]) >= net_filter_count
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

        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)
        self.hard_positives: bool = dataset_options.get("hard_positives", False)
        self.target_model_name: str = dataset_options.get(
            "target_model_name", "japanese-splade-base-v1-mmarco-only"
        )
        self.query_column_name: str = dataset_options.get("query_column_name", "anc")
        self.doc_column_name: str = dataset_options.get("doc_column_name", "text")
        dataset_name = dataset_options.get("dataset_name", "mmarco")
        logger.info(f"Initializing {dataset_name} hard_negative dataset")
        logger.info(f"binarize_label: {self.binarize_label}")
        logger.info(f"hard_positives: {self.hard_positives}")
        logger.info(f"target_model_name: {self.target_model_name}")
        logger.info(f"query_column_name: {self.query_column_name}")
        logger.info(f"doc_column_name: {self.doc_column_name}")

        query_ds_name = f"{dataset_name}-dataset"
        collection_ds_name = f"{dataset_name}-collection"

        neg_score_th = dataset_options.get("neg_score_th", NEG_SCORE_TH)
        pos_score_th = dataset_options.get("pos_score_th", POS_SCORE_TH)
        neg_pos_thcore_th = dataset_options.get("neg_pos_thcore_th", NEG_POS_SCORE_TH)
        net_filter_count = dataset_options.get("net_filter_count", NEG_FILTER_COUNT)

        self.top_100_sampling_count = dataset_options.get(
            "top_100_sampling_count", TOP_100_SAMPLING_COUNT
        )

        ds = load_dataset(HADR_NEGATIVE_SCORE_DS, query_ds_name, split=DS_SPIT)
        self.collection_ds = load_dataset(
            HADR_NEGATIVE_SCORE_DS, collection_ds_name, split=DS_SPIT
        )
        ds = ds.map(
            _map_score_with_hard_positives,
            num_proc=11,  # type: ignore
            fn_kwargs={
                "neg_score_th": neg_score_th,
                "pos_score_th": pos_score_th,
                "neg_pos_score_th": neg_pos_thcore_th,
                "hard_positives": self.hard_positives,
                "target_model_name": self.target_model_name,
            },
        )  # type: ignore
        ds = ds.filter(
            _filter_score, num_proc=11, fn_kwargs={"net_filter_count": net_filter_count}
        )  # type: ignore
        logger.info(f"Filtered dataset size: {len(ds)}")

        aug_factor = dataset_options.get("aug_factor", 1.0)
        n = int(dataset_options.get("n", 0))
        if aug_factor != 1.0:
            n = int(len(ds) * (aug_factor))
            logging.info(
                f"Augmenting dataset with factor aug_factor={aug_factor}, n={n}"
            )
        if n > len(ds):
            logger.info(f"Expanding dataset from {len(ds)} to {n}")
            ds_expand = []
            c = n // len(ds)
            r = n % len(ds)
            for _ in range(c):
                ds_expand.append(ds.shuffle(seed=42))
            ds_expand.append(ds.shuffle(seed=42).select(range(r)))  # type: ignore
            ds = concatenate_datasets(ds_expand)
            assert len(ds) == n
        elif n > 0:
            logger.info(f"Shuffling and selecting first {n} samples from dataset")
            ds = ds.shuffle(seed=42).select(range(n))  # type: ignore

        super().__init__(args, tokenizer, ds)  # type: ignore

    def get_collection_text(self, doc_id: int) -> str:
        text = self.collection_ds[doc_id][self.doc_column_name]  # type: ignore
        return text  # type: ignore

    def __getitem__(self, item) -> list[dict]:
        group_size = self.args.train_group_size
        query = self.dataset[item][self.query_column_name]

        pos_ids = self.dataset[item]["pos"]
        pos_ids_score = self.dataset[item]["pos.score"]

        neg_ids_top100 = self.dataset[item]["neg.top100"]
        neg_ids_score_top100 = self.dataset[item]["neg.score.top100"]

        # N をneg_ids_top100からrandom sampling
        top_100_count = self.top_100_sampling_count
        if len(neg_ids_top100) < top_100_count:
            top_100_count = len(neg_ids_top100)
        neg_ids_top100_sampled = random.sample(neg_ids_top100, top_100_count)
        neg_ids_score_top100_sampled = [
            neg_ids_score_top100[neg_ids_top100.index(id_)]
            for id_ in neg_ids_top100_sampled
        ]

        other_100_count = group_size - top_100_count - 1
        neg_ids_other100 = self.dataset[item]["neg.other100"]
        neg_ids_score_other100 = self.dataset[item]["neg.score.other100"]
        if len(neg_ids_other100) < other_100_count:
            other_100_count = len(neg_ids_other100)
        neg_ids_other100_sampled = random.sample(neg_ids_other100, other_100_count)
        neg_ids_score_other100_sampled = [
            neg_ids_score_other100[neg_ids_other100.index(id_)]
            for id_ in neg_ids_other100_sampled
        ]

        neg_ids = neg_ids_top100_sampled + neg_ids_other100_sampled
        neg_ids_score = neg_ids_score_top100_sampled + neg_ids_score_other100_sampled

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
