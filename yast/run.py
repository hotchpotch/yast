"""
Code adapted from: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/run.py
License: MIT License
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .arguments import (
    DataArguments,
    ModelArguments,
    RunArguments,
    SpladeTrainingArguments,
)
from .data import (
    GroupCollator,
    create_dateset_from_args,
)
from .modeling import Splade
from .trainer import SpladeTrainer
from .utils import seed_everything

logger = logging.getLogger(__name__)


def _setup_wandb():
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "splade"


def main():
    _setup_wandb()

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, SpladeTrainingArguments, RunArguments)  # type: ignore
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, run_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and (
        sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")
    ):
        model_args, data_args, training_args, run_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, run_args = (
            parser.parse_args_into_dataclasses()
        )

    seed_everything(training_args.seed)

    training_args.remove_unused_columns = False  # override

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        use_fast=False,
    )

    model = Splade.from_pretrained(
        model_args,
        model_args.model_name_or_path,
    )

    train_dataset = create_dateset_from_args(data_args, tokenizer)
    trainer = SpladeTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    last_checkpoint = None
    if training_args.resume_from_checkpoint or os.environ.get(
        "RESUME_FROM_CHECKPOINT", False
    ):
        training_args.resume_from_checkpoint = True
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info("[RESUME] last_checkpoint: %s", last_checkpoint)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    if run_args.remove_checkpoints:
        logger.info("Remove checkpoints")
        # remove checkpoints
        for dir in Path(training_args.output_dir).glob("checkpoint-*"):
            if dir.is_dir():
                shutil.rmtree(dir)


if __name__ == "__main__":
    main()
