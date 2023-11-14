import os
import logging
from typing import Union, Optional, Sequence, NoReturn

from datasets import Dataset
from transformers import (
    set_seed,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast, PretrainedConfig, default_data_collator, DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint

from classification_datasets.builder import DatasetBuilder, ClassificationDataset
from classification_datasets.labels import LabelInfo
from classification_datasets.text_tokenizers import SpacyTokenizer

from utils import (
    setup_logging,
    log_training_args
)

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/obi0/phi/ehr_projects/phenotype_classification/cache/huggingface/datasets/'


class Classifier(object):
    """
    The SequenceTagger class can be used to train sequence tagging models, evaluate models, and
    run predictions. The class can also be used to de-identify and augment text.
    TODO: Add a more detailed description
    """

    def __init__(
            self,
            text_tokenizer: Optional[SpacyTokenizer],
            config: PretrainedConfig,
            subword_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            model: PreTrainedModel,
            label_list: Sequence[str],
            ignore_labels: Sequence[str],
            fp16: bool,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = None,
            is_split_into_words: bool = False,
            text_column_name: str = 'text',
            label_column_name: str = 'label',
            seed: int = 19,
            classification_threshold=None,
            optimal_threshold=None
    ):
        """
        Initialize the variables and load the relevant objects.

        Args:
            text_tokenizer (SpacyTokenizer): The SpacyTokenizer object for word tokenization of text.
            config (PretrainedConfig): The HuggingFace model config object.
            subword_tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The Huggingface tokenizer object.
            model (PreTrainedModel): The HuggingFace model object.
            label_list (Sequence[str]):
            fp16 (bool): Whether to use mixed-precision.
            pad_to_max_length (bool, defaults to `True`): Whether to pad all samples to `max_seq_length`. If False,
            will pad the samples dynamically when batching to the maximum length in the batch.
            max_seq_length (Optional[int], defaults to `None`): The maximum total input sequence length after
            tokenization. Sequences longer than this will be truncated.
            seed (int, defaults to `41): Reproducible seed.
        """

        # Set seed before initializing model.
        set_seed(seed)
        self._seed = seed
        self._classification_threshold = classification_threshold
        self._optimal_threshold = optimal_threshold

        # Initialize the label list
        self._label_list = label_list
        self._ignore_labels = ignore_labels

        # This object will have label information
        if self._label_list is not None:
            self._label_info = LabelInfo(label_list=self._label_list)
        else:
            # TODO: Fix this
            self._label_info = None

        # Initialize variables that will be set in the set_trainer function to None
        self._trainer = None
        self.__train_samples = None
        self.__eval_samples = None

        # Initialize the tokenizers (HuggingFace and SpaCy), Huggingface config, Huggingface model and data collator
        self._text_tokenizer = text_tokenizer
        self._config = config
        self._problem_type = self._config.problem_type
        self._subword_tokenizer = subword_tokenizer
        self._model = model

        self._multi_class = self._check_multi_class()
        self._multi_label = self._check_multi_label()

        self._data_collator = self.get_data_collator(pad_to_max_length=pad_to_max_length, fp16=fp16)

        # Set the label2id and id2label mapping of the HuggingFace config object
        if self._label_info is not None:
            self._model.config.label2id = self._label_info.get_label_to_id()
            self._model.config.id2label = self._label_info.get_id_to_label()
            self._label_list = self._label_info.get_label_list()
        else:
            self._label_list = list(self._config.label2id)

        # Set the label_to_id
        self.label_to_id = self._model.config.label2id

        # Set the TextTransform
        self._classification_transform = self.get_classification_transform(
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            is_split_into_words=is_split_into_words
        )

        self._post_process = self.get_post_process()

    def _check_multi_class(self):
        raise NotImplementedError('Implement in subclass')

    def _check_multi_label(self):
        raise NotImplementedError('Implement in subclass')

    def get_post_process(self):
        raise NotImplementedError('Implement in subclass')

    def get_label_list(self) -> Sequence[str]:
        """
        Return the NER label list.

        Returns:
            _label_list (Sequence[str]): The NER label list.
        """

        return self._label_list

    def get_data_collator(
            self,
            pad_to_max_length,
            fp16: bool
    ) -> Optional[Union[default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification]]:
        """
        Get the data collator object.

        Args:
            pad_to_max_length (bool, defaults to `True`): Whether to pad all samples to `max_seq_length`. If False,
            will pad the samples dynamically when batching to the maximum length in the batch.
            fp16 (bool): Whether to use mixed precision training.

        Returns:
            (DataCollatorForTokenClassification): The data collator object.
        """
        raise NotImplementedError('Implement in the subclass')

    def get_classification_transform(
            self,
            pad_to_max_length,
            max_seq_length,
            text_column_name,
            label_column_name,
            is_split_into_words
    ):

        raise NotImplementedError('Implement in the subclass')

    def get_train_dataset(
            self,
            train_file: Optional[str],
            training_args: TrainingArguments,
            train_on_fly: bool = True,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            max_train_samples: Optional[int] = None
    ) -> Optional[Dataset]:
        """
        Get the training dataset.

        Args:
            train_file (Optional[str]): The file that contains the training data.
            training_args (TrainingArguments): The HuggingFace training arguments object.
            train_on_fly (bool, defaults to `True`): Option to return text transformations on the fly (e.g. tokenization
            and augmentation etc.).
            preprocessing_num_workers (Optional[int], defaults to `None`): The number of processes to use for the
            preprocessing.
            overwrite_cache (bool, defaults to `False`): Overwrite the cached training sets.
            max_train_samples (Optional[int], defaults to `None`): For debugging purposes or quicker training, truncate
            the number of training examples to this value if set.

        Returns:
            train_dataset (Optional[Dataset]): The training dataset if the train file is not None else None.
        """

        # Return none if train file is not passed
        if train_file is None:
            return None

        # Create a sequence dataset object to load the training dataset
        raw_text_datasets = DatasetBuilder(
            train_file=train_file,
            validation_file=None,
            test_file=None,
            cache_dir=HF_DATASETS_CACHE
        ).raw_datasets
        classification_datasets = ClassificationDataset(
            text_datasets=raw_text_datasets,
        )
        # Get the train split of the dataset
        train_dataset = classification_datasets.get_train_dataset(max_train_samples=max_train_samples)

        # Apply any text transformations on the fly or initially
        if train_on_fly:
            train_dataset.set_transform(self._classification_transform.encode_train)
        else:
            remove_columns = train_dataset.column_names
            # Run the tokenization process
            with training_args.main_process_first(desc="Train dataset map tokenization"):
                train_dataset = train_dataset.map(
                    self._classification_transform.encode_train,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    remove_columns=remove_columns,
                    desc="Running text transformations on train dataset",
                )
        return train_dataset.shuffle(seed=self._seed)

    def get_validation_dataset(
            self,
            validation_file: Optional[str],
            training_args: TrainingArguments,
            validation_on_fly: bool = False,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            max_eval_samples: Optional[int] = None
    ):
        """
        Get the validation dataset.

        Args:
            validation_file (Optional[str]): The file that contains the validation data.
            training_args (TrainingArguments): The HuggingFace training arguments object.
            validation_on_fly (bool, defaults to `True`): Option to return text transformations on the fly
            (e.g. tokenization and augmentation etc.).
            preprocessing_num_workers (Optional[int], defaults to `None`): The number of processes to use for the
            preprocessing.
            overwrite_cache (bool, defaults to `False`): Overwrite the cached validation sets.
            max_eval_samples (Optional[int], defaults to `None`): For debugging purposes or quicker training, truncate
            the number of validation examples to this value if set.

        Returns:
            eval_dataset (Optional[Dataset]): The validation dataset if the validation file is not None else None.
        """

        # Return none if validation file is not passed
        if validation_file is None:
            return None

        # Create a sequence dataset object to load the validation dataset
        raw_text_datasets = DatasetBuilder(
            train_file=None,
            validation_file=validation_file,
            test_file=None,
            cache_dir=HF_DATASETS_CACHE
        ).raw_datasets
        classification_datasets = ClassificationDataset(
            text_datasets=raw_text_datasets,
        )
        # Get the validation split of the dataset
        eval_dataset = classification_datasets.get_eval_dataset(max_eval_samples=max_eval_samples)

        # Apply any text transformations on the fly or initially
        if validation_on_fly:
            eval_dataset.set_transform(self._classification_transform.encode_validation)
        else:
            remove_columns = eval_dataset.column_names
            with training_args.main_process_first(desc="Validation dataset map tokenization"):
                eval_dataset = eval_dataset.map(
                    self._classification_transform.encode_validation,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    remove_columns=remove_columns,
                    desc="Running text transformations on validation dataset",
                )
        return eval_dataset

    def get_test_dataset(
            self,
            test_file: Optional[str],
            training_args: TrainingArguments,
            test_on_fly: bool = False,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            max_predict_samples: Optional[int] = None
    ):
        """
        Get the test dataset.

        Args:
            test_file (Optional[str]): The file that contains the test data.
            training_args (TrainingArguments): The HuggingFace training arguments object.
            test_on_fly (bool, defaults to `True`): Option to return text transformations on the fly
            (e.g. tokenization and augmentation etc.).
            preprocessing_num_workers (Optional[int], defaults to `None`): The number of processes to use for the
            preprocessing.
            overwrite_cache (bool, defaults to `False`): Overwrite the cached validation sets.
            max_predict_samples (Optional[int], defaults to `None`): For debugging purposes or quicker training,
            truncate the number of test examples to this value if set.

        Returns:
            test_dataset (Optional[Dataset]): The test dataset if the test file is not None else None.
        """

        # Return none if test file is not passed
        if test_file is None:
            return None

        # Create a sequence dataset object to load the test dataset
        raw_text_datasets = DatasetBuilder(
            train_file=None,
            validation_file=None,
            test_file=test_file,
            cache_dir=HF_DATASETS_CACHE
        ).raw_datasets
        classification_datasets = ClassificationDataset(
            text_datasets=raw_text_datasets,
        )
        test_dataset = classification_datasets.get_test_dataset(max_test_samples=max_predict_samples)

        # Apply any text transformations on the fly or initially
        if test_on_fly:
            test_dataset.set_transform(self._classification_transform.encode_test)
        else:
            with training_args.main_process_first(desc="Test dataset map tokenization"):
                test_dataset = test_dataset.map(
                    self._classification_transform.encode_test,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc="Running text transformations on test dataset",
                )
        return test_dataset

    def set_trainer(
            self,
            training_args: TrainingArguments,
            train_dataset: Optional[Dataset],
            eval_dataset: Optional[Dataset],
            do_train: bool,
            do_eval: bool,
            do_predict: bool,
            train_on_fly: bool = True
    ) -> NoReturn:
        """
        Set the HuggingFace trainer object. If the models are being trained/evaluated, the
        function expects the train and validation datasets to be passed. Based on the
        parameters passed, the HuggingFace trainer is set and can be used to train,
        evaluate and run predictions on the model.

        Args:
            training_args (TrainingArguments): The HuggingFace training arguments object.
            train_dataset (Optional[Dataset]): The training dataset when do_train is set to True.
            eval_dataset (Optional[Dataset]): The validation dataset when do_eval is set to True.
            do_train (bool): Whether to train the model.
            do_eval (bool): Whether to evaluate the model on the validation dataset.
            do_predict (bool): Whether to run predictions on the test dataset.
            train_on_fly (bool, defaults to `True`): Option to return text transformations on the fly (e.g. tokenization
            and augmentation etc.).
        """

        # Compute metrics will be None when do_eval is False
        compute_metrics = None

        # Set the remove columns parameter based on the dataset inputs
        # We will remove columns based on whether we're running things
        # on the fly or not
        # TODO Change these conditions based on on-the-fly parameters
        if do_predict and not train_on_fly:
            # Remove unused columns from test set - running forward pass
            training_args.remove_unused_columns = True
        else:
            # Setting to False so that unused columns can be used for the on the fly transformations
            training_args.remove_unused_columns = False

        # Check if train dataset is passed when do_train is True
        if do_train and train_dataset is None:
            raise ValueError("--do_train requires a train dataset")

        # Check if validation dataset is passed when do_eval is True
        # Set the metrics function for validation dataset
        if do_eval:
            if eval_dataset is None:
                raise ValueError("--do_eval requires a validation dataset")
            compute_metrics = self.get_metrics()

        # Initialize our Trainer
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._subword_tokenizer,
            data_collator=self._data_collator,
            compute_metrics=compute_metrics,
        )

        # Set the number of training and evaluation samples
        self.__train_samples = len(train_dataset) if train_dataset is not None else None
        self.__eval_samples = len(eval_dataset) if eval_dataset is not None else None

    def get_metrics(self):
        raise NotImplementedError('Implement in the subclass')

    def run_train(self, resume_from_checkpoint, last_checkpoint):
        if self.__train_samples is None:
            return None
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self._trainer.train(resume_from_checkpoint=checkpoint)
        self._trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = self.__train_samples

        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)
        self._trainer.save_state()
        return metrics

    def run_eval(self):
        if self.__eval_samples is None:
            return None
        logging.info("*** Evaluate ***")
        metrics = self._trainer.evaluate()

        metrics["eval_samples"] = self.__eval_samples
        self._trainer.log_metrics("eval", metrics)
        self._trainer.save_metrics("eval", metrics)
        return metrics

    def run_predict(self, test_dataset):
        if test_dataset is None:
            return None
        return self._trainer.predict(test_dataset)

    @staticmethod
    def log_args(training_args: TrainingArguments) -> NoReturn:
        """
        Log the training arguments.

        Args:
            training_args (TrainingArguments): The HuggingFace training arguments.
        """

        # Setup logging
        setup_logging(logger=logger, log_level=training_args.get_process_log_level())

        # Log the training arguments
        log_training_args(training_args=training_args)

    @staticmethod
    def get_checkpoint(
            output_dir: str,
            resume_from_checkpoint: Union[str, bool],
            do_train: bool, overwrite_output_dir: bool
    ):
        last_checkpoint = None
        if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and resume_from_checkpoint is None:
                logging.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        return last_checkpoint
