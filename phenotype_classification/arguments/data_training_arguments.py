from dataclasses import field, dataclass
from typing import Optional


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_on_fly: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to run text transformations (e.g tokenization) on the fly"}
    )
    validation_on_fly: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to run text transformations (e.g tokenization) on the fly"}
    )
    test_on_fly: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to run text transformations (e.g tokenization) on the fly"}
    )
    label_list: Optional[str] = field(
        default=None,
        metadata={"help": "The list of labels"}
    )
    ignore_labels: Optional[str] = field(
        default=None,
        metadata={"help": "The list of labels"}
    )
    prompt_label_list: Optional[str] = field(
        default=None,
        metadata={"help": "The list of prompt labels"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column that contains the text data"}
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column that contains the label data"}
    )
    note_id_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column that contains the note ids"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            if type(self.train_file) == list:
                extension = self.train_file[0].split(".")[-1]
            else:
                extension = self.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "parquet", "zst", "jsonl"]:
                raise ValueError("`train_file` should be a csv, a parquet, a json or a txt file.")
        if self.validation_file is not None:
            if type(self.validation_file) == list:
                extension = self.validation_file[0].split(".")[-1]
            else:
                extension = self.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "parquet", "zst", "jsonl"]:
                raise ValueError("`validation_file` should be a csv, a parquet, a json or a txt file.")
