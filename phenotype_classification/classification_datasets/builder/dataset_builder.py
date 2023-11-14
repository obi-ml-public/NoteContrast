from datasets import load_dataset, DatasetDict
from typing import Optional, Dict, NoReturn

from transformers import PreTrainedTokenizerFast, TrainingArguments


class DatasetBuilder(object):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(
            self,
            train_file: Optional[str] = None,
            validation_file: Optional[str] = None,
            test_file: Optional[str] = None,
            cache_dir: Optional[str] = None,
            use_auth_token: bool = False,
    ):
        """
        Create a Huggingface dataset object. Read the train and validation files.
        The dataset object can be tokenized with the tokenize_dataset function.
        Handles tokenization, padding, truncation etc. of the dataset.

        Args:
            train_file (str, defaults to `None`): The input training data file
            validation_file (str, defaults to `None`): An optional input evaluation data file to evaluate the
            perplexity/accuracy on
            test_file (str, defaults to `None`): An optional input file to run the forward pass on
            cache_dir (Optional[str], defaults to `None`): Where do you want to store the pretrained models/data
            downloaded from huggingface.co
            use_auth_token (bool, defaults to `False`): Will use the token generated when running
            `transformers-cli login` (necessary to use this script with private models).
        """

        # Store the train and validation file paths
        data_files = {}
        extension = None
        if train_file is not None:
            data_files["train"] = train_file
            if type(train_file) == list:
                extension = train_file[0].split(".")[-1]
            else:
                extension = train_file.split(".")[-1]
        if validation_file is not None:
            data_files["validation"] = validation_file
            if type(validation_file) == list:
                extension = validation_file[0].split(".")[-1]
            else:
                extension = validation_file.split(".")[-1]
        if test_file is not None:
            data_files["test"] = test_file
            if type(test_file) == list:
                extension = test_file[0].split(".")[-1]
            else:
                extension = test_file.split(".")[-1]
        if extension == 'jsonl' or extension == 'zst':
            extension = 'json'
        # Load the dataset object
        self.__load_dataset(
            data_files=data_files, extension=extension, cache_dir=cache_dir, use_auth_token=use_auth_token
        )

    def __load_dataset(
            self,
            data_files: Dict[str, str],
            extension: str,
            cache_dir: Optional[str] = None,
            use_auth_token: bool = False
    ) -> NoReturn:
        """
        Read the train and validation files and load the dataset object

        Args:
            data_files (Dict[str, str]): Mapping between dataset split, and it's path
            extension (str): The file extension of the paths the data is stored in
            cache_dir (Optional[str], defaults to `None`): Where do you want to store the pretrained models/data
            downloaded from huggingface.co
            use_auth_token (bool, defaults to `False`): Will use the token generated when running
            `transformers-cli login` (necessary to use this script with private models).
        """

        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        self.raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )

    def tokenize_dataset(
            self,
            tokenizer: PreTrainedTokenizerFast,
            training_args: TrainingArguments,
            truncation: bool = True,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = 514,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
    ) -> DatasetDict:
        """
        Tokenize the dataset object. Read in the column that contains the input text and tokenize the
        text with the specified tokenizer. Handles the addition and truncation of position ids, if the
        positions ids are not None.

        Args:
            tokenizer (PreTrainedTokenizerFast): Tokenizer object used to tokenize the input text
            training_args (TrainingArguments): The training arguments from huggingface
            truncation (bool, defaults to `True`): Truncate the text to the model max length
            pad_to_max_length (bool, defaults to `False`): Whether to pad all samples to `max_seq_length`.
            If False, will pad the samples dynamically when batching to the maximum length in the batch.
            max_seq_length (Optional[int], defaults to `514`): The maximum total input sequence length after
            tokenization. Sequences longer than this will be truncated.
            preprocessing_num_workers (Optional[int], defaults to `None`): The number of processes to use for
            the preprocessing.
            overwrite_cache (bool, defaults to `False`): Overwrite the cached training and evaluation sets

        Returns:
            tokenized_datasets (DatasetDict): The dataset object that contains the tokenized train and validation splits
        """

        raise NotImplementedError('This function needs to be implemented by the subclass')
