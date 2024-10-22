"""
JMTEB retrieval データセットから、スパース性を測るためクエリとドキュメントのL0を出力
"""

import argparse
from typing import Dict, List, cast

import datasets
import numpy as np
from yasem import SpladeEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure L0 sparsity for multiple models on JMTEB dataset."
    )
    parser.add_argument(
        "-m",
        "--model_names",
        type=str,
        nargs="+",
        default=["hotchpotch/japanese-splade-base-v1"],
        help="List of model names to evaluate.",
    )
    return parser.parse_args()


def map_corpus_text(example):
    title = example.get("title", "")
    text = example.get("text", "")
    if title:
        text = title + " " + text
    return {"text": text}


def main():
    args = parse_args()
    model_names: List[str] = args.model_names

    JMTEB_TARGETS = [
        "jaqket",
        "mrtydi",
        "jagovfaqs_22k",
        "nlp_journal_title_abs",
        "nlp_journal_title_intro",
        "nlp_journal_abs_intro",
    ]
    JMTEB_QUERY_SPLIT_TARGET = "test"
    QUERY_MAX_SAMPLE_SIZE = 1000
    CORPUS_MAX_SAMPLE_SIZE = 1000

    # Initialize result dictionary with keys as 'target-query' and 'target-docs'
    result_dict: Dict[str, Dict[str, float]] = {}
    for target in JMTEB_TARGETS:
        result_dict[f"{target}-query"] = {}
        result_dict[f"{target}-docs"] = {}

    for model_name in model_names:
        print(f"Processing Model: {model_name}")
        embedder = SpladeEmbedder(model_name)

        for target in JMTEB_TARGETS:
            print(f"  Processing Target: {target}")
            # Load query dataset
            target_query_ds = datasets.load_dataset(
                "sbintuitions/JMTEB",
                name=f"{target}-query",
                split=JMTEB_QUERY_SPLIT_TARGET,
                trust_remote_code=True,
            )
            target_query_ds = cast(datasets.Dataset, target_query_ds)

            target_query_ds = target_query_ds.select(
                range(min(QUERY_MAX_SAMPLE_SIZE, len(target_query_ds)))
            )

            # Load corpus dataset
            target_corpus_ds = datasets.load_dataset(
                "sbintuitions/JMTEB",
                name=f"{target}-corpus",
                split="corpus",
                trust_remote_code=True,
            )
            target_corpus_ds = cast(datasets.Dataset, target_corpus_ds)
            target_corpus_ds = target_corpus_ds.select(
                range(min(CORPUS_MAX_SAMPLE_SIZE, len(target_corpus_ds)))
            )
            target_corpus_ds = target_corpus_ds.map(map_corpus_text, num_proc=4)

            print(f"    Query size: {len(target_query_ds)}")
            print(f"    Docs size: {len(target_corpus_ds)}")

            # Encode queries and documents
            query_vectors = embedder.encode(
                target_query_ds["query"],
                convert_to_csr_matrix=True,
                show_progress_bar=True,
            )
            corpus_vectors = embedder.encode(
                target_corpus_ds["text"],
                convert_to_csr_matrix=True,
                show_progress_bar=True,
            )

            # Calculate L0
            query_L0 = np.mean(np.diff(query_vectors.indptr))  # type:ignore
            docs_L0 = np.mean(np.diff(corpus_vectors.indptr))  # type:ignore

            print(f"    {target} Queries L0: {query_L0}")
            print(f"    {target} Docs L0: {docs_L0}")

            # Store results
            result_dict[f"{target}-query"][model_name] = query_L0  # type:ignore
            result_dict[f"{target}-docs"][model_name] = docs_L0  # type:ignore

    # Generate Markdown Table
    print("\n## L0 Sparsity\n")
    header = ["Target"] + model_names
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for target_key in sorted(result_dict.keys()):
        row = [target_key]
        for model in model_names:
            l0_value = result_dict[target_key].get(model, 0.0)
            row.append(f"{l0_value:.1f}")
        print("| " + " | ".join(row) + " |")

    # Generate CSV Output
    print("\n## CSV Output\n")
    print("Target," + ",".join(model_names))
    for target_key in sorted(result_dict.keys()):
        row = [target_key] + [
            f"{result_dict[target_key].get(model, 0.0):.1f}" for model in model_names
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
