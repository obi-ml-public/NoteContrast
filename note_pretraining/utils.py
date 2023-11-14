import logging
import sys
from typing import NoReturn, Sequence, List

import datasets
import transformers
from transformers import (
    TrainingArguments
)


def setup_logging(logger, log_level: int) -> NoReturn:
    """
    Function sets up the log level and format.

    Args:
        logger (): TO-DO
        log_level (int): The numeric value of the log level.

    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def log_training_args(training_args: TrainingArguments) -> NoReturn:
    """
    Log the training argument values.

    Args:
        training_args (TrainingArguments): The training arguments.

    """
    # Log on each process the small summary:
    logging.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logging.info(f"Training/evaluation parameters {training_args}")


def unpack_nested_list(nested_list: Sequence[Sequence[str]]) -> List[str]:
    """
    Use this function to unpack a nested list.

    Args:
        nested_list (Sequence[Sequence[str]]): A nested list.

    Returns:
        (List[str]): Flattened list.

    """
    return [inner for nested in nested_list for inner in nested]
