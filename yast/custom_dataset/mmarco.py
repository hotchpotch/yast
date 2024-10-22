import logging

import joblib
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForSpladeTraining

logger = logging.getLogger(__name__)

MMARCO_DATASET = "unicamp-dl/mmarco"
HADR_NEGATIVE_SCORE_DS = "hotchpotch/mmarco-hard-negatives-reranker-score"

NEG_SCORE_TH = 0.3
POS_SCORE_TH = 0.7
NEG_FILTER_COUNT = 8


def _map_filter_score(example, neg_score_th: float, pos_score_th: float):
    neg_score: list[float] = example["neg.score"]
    neg_score_filtered_index = [
        i for i, score in enumerate(neg_score) if score < neg_score_th
    ]
    # same pos_score
    pos_score = example["pos.score"]
    pos_score_filtered_index = [
        i for i, score in enumerate(pos_score) if score > pos_score_th
    ]
    return {
        **example,
        "neg.score": [neg_score[i] for i in neg_score_filtered_index],
        "neg": [example["neg"][i] for i in neg_score_filtered_index],
        "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        "pos": [example["pos"][i] for i in pos_score_filtered_index],
    }


def _filter_score(example, net_filter_count: int):
    # neg のカウントがN以上で、pos のカウントが1以上のものを返す
    return len(example["neg"]) >= net_filter_count and len(example["pos"]) >= 1


class MMarcoHardNegatives(DatasetForSpladeTraining):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        logger.info("Initializing MMarcoHardNegatives dataset")

        self.query_max_len = args.dataset_options.get("query_max_len", 256)
        self.doc_max_len = args.dataset_options.get("doc_max_len", 1024)
        train_data = args.train_data

        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)

        lang = train_data["lang"]
        reranker_name = train_data["reranker"]
        neg_score_th = train_data.get("neg_score_th", NEG_SCORE_TH)
        pos_score_th = train_data.get("pos_score_th", POS_SCORE_TH)
        net_filter_count = train_data.get("net_filter_count", NEG_FILTER_COUNT)
        subset = f"{lang}_{reranker_name}"

        mapping = f"mappings/{lang}_joblib.pkl.gz"

        logger.info(f"Downloading mapping file from Hugging Face Hub: {mapping}")
        mapping_file = hf_hub_download(
            repo_type="dataset", repo_id=HADR_NEGATIVE_SCORE_DS, filename=mapping
        )

        logger.info(f"Loading mapping file: {mapping_file}")
        index_mapping_dict = joblib.load(mapping_file)

        self.query_id_dict = index_mapping_dict["query_id_dict"]
        self.collection_id_dict = index_mapping_dict["collection_id_dict"]

        logger.info(f"Loading queries dataset for language: {lang}")
        self.queries_ds = load_dataset(
            MMARCO_DATASET,
            "queries-" + lang,
            split="train",
            trust_remote_code=True,
        )
        logger.info(f"Loading collection dataset for language: {lang}")
        self.collection_ds = load_dataset(
            MMARCO_DATASET,
            "collection-" + lang,
            split="collection",
            trust_remote_code=True,
        )
        logger.info(f"Loading hard negatives dataset subset: {subset}")
        ds = load_dataset(HADR_NEGATIVE_SCORE_DS, subset, split="train")
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

    def get_query_text(self, query_id: int) -> str:
        idx = self.query_id_dict[query_id]
        return self.queries_ds[idx]["text"][0 : self.query_max_len]  # type: ignore

    def get_collection_text(self, doc_id: int) -> str:
        idx = self.collection_id_dict[doc_id]
        return self.collection_ds[idx]["text"][0 : self.doc_max_len]  # type: ignore

    def __getitem__(self, item) -> list[dict]:
        qid = self.dataset[item]["qid"]
        query = self.get_query_text(qid)
        pos_ids = self.dataset[item]["pos"]
        pos_ids_score = self.dataset[item]["pos.score"]
        neg_ids = self.dataset[item]["neg"]
        neg_ids_score = self.dataset[item]["neg.score"]

        pos_texts = [self.get_collection_text(pos_id) for pos_id in pos_ids]
        neg_texts = [self.get_collection_text(neg_id) for neg_id in neg_ids]

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
