import os
import logging
from typing import Union, Optional, NoReturn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from datasets import Dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint

from models import ICDRobertaForMaskedLM
from arguments import ModelArguments, DataTrainingArguments
from pre_train_datasets.builder import DatasetBuilder, PreTrainDataset
from pre_train_datasets.data_collator import ICDDataCollatorForLanguageModeling
from pre_train_datasets.text_transformations import MLMTransform, Transform

from model_helpers import ModelHelpers
from pre_train_metrics import MLMMetrics

from utils import (
    setup_logging,
    log_training_args
)

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/<path>/phi/ehr_projects/icd_embeddings/cache/huggingface/datasets/'


class MLMPreTrainer(object):
    """
    The SequenceTagger class can be used to train sequence tagging models, evaluate models, and
    run predictions. The class can also be used to de-identify and augment text.
    TODO: Add a more detailed description
    """

    def __init__(
            self,
            training_args,
            config: PretrainedConfig,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            model: ICDRobertaForMaskedLM,
            fp16: bool,
            mlm_probability: float = 0.20,
            pad_to_max_length: bool = False,
            max_seq_length: Optional[int] = 512,
            text_column_name: str = 'text',
            position_ids_column_name: Optional[str] = None,
            token_type_ids_column_name: Optional[str] = None,
            seed: int = 41
    ):
        """
        Initialize the variables and load the relevant objects.

        Args:
        """

        # Set seed before initializing model.
        set_seed(seed)

        # Initialize variables that will be set in the set_trainer function to None
        self._training_args = training_args
        self._trainer = None
        self.__train_samples = None
        self.__eval_samples = None

        # Initialize the tokenizers (HuggingFace and SpaCy), Huggingface config, Huggingface model and data collator
        self._config = config
        self._tokenizer = tokenizer
        self._model = model

        self._data_collator = self.get_data_collator(
            pad_to_max_length=pad_to_max_length,
            fp16=fp16,
            mlm_probability=mlm_probability,
        )

        # Set the TextTransform
        self._mlm_transform = self.get_mlm_transform(
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            text_column_name=text_column_name,
            position_ids_column_name=position_ids_column_name,
            token_type_ids_column_name=token_type_ids_column_name
        )

    def get_data_collator(
            self,
            pad_to_max_length,
            fp16: bool,
            mlm_probability,
    ) -> ICDDataCollatorForLanguageModeling:
        """
        Get the data collator object.
        """
        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer,
        # so we change it if we already did the padding.
        # Load data collator
        pad_to_multiple_of_8 = fp16 and not pad_to_max_length
        return ICDDataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    def get_mlm_transform(
            self,
            pad_to_max_length,
            max_seq_length,
            text_column_name,
            position_ids_column_name,
            token_type_ids_column_name
    ):

        transform = Transform(
            tokenizer=self._tokenizer,
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            truncation=True
        )

        return MLMTransform(
            transform=transform,
            text_column_name=text_column_name,
            position_ids_column_name=position_ids_column_name,
            token_type_ids_column_name=token_type_ids_column_name,
        )

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
        pre_train_datasets = PreTrainDataset(
            text_datasets=raw_text_datasets,
        )
        # Get the train split of the dataset
        train_dataset = pre_train_datasets.get_train_dataset(max_train_samples=max_train_samples)

        # Apply any text transformations on the fly or initially
        if train_on_fly:
            train_dataset.set_transform(self._mlm_transform.encode_train)
        else:
            remove_columns = train_dataset.column_names
            # Run the tokenization process
            with training_args.main_process_first(desc="Train dataset map tokenization"):
                train_dataset = train_dataset.map(
                    self._mlm_transform.encode_train,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    remove_columns=remove_columns,
                    desc="Running text transformations on train dataset",
                )
        return train_dataset.shuffle()

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
        pre_train_datasets = PreTrainDataset(
            text_datasets=raw_text_datasets,
        )
        # Get the validation split of the dataset
        eval_dataset = pre_train_datasets.get_eval_dataset(max_eval_samples=max_eval_samples)

        # Apply any text transformations on the fly or initially
        if validation_on_fly:
            eval_dataset.set_transform(self._mlm_transform.encode_validation)
        else:
            remove_columns = eval_dataset.column_names
            with training_args.main_process_first(desc="Validation dataset map tokenization"):
                eval_dataset = eval_dataset.map(
                    self._mlm_transform.encode_validation,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    remove_columns=remove_columns,
                    desc="Running text transformations on validation dataset",
                )
        return eval_dataset.shuffle()

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
        pre_train_datasets = PreTrainDataset(
            text_datasets=raw_text_datasets,
        )
        test_dataset = pre_train_datasets.get_test_dataset(max_test_samples=max_predict_samples)

        # Apply any text transformations on the fly or initially
        if test_on_fly:
            test_dataset.set_transform(self._mlm_transform.encode_test)
        else:
            with training_args.main_process_first(desc="Test dataset map tokenization"):
                test_dataset = test_dataset.map(
                    self._mlm_transform.encode_test,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc="Running text transformations on test dataset",
                )
        return test_dataset

    def set_trainer(
            self,
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
            train_dataset (Optional[Dataset]): The training dataset, when do_train is set to True.
            eval_dataset (Optional[Dataset]): The validation dataset, when  do_eval is set to True.
            do_train (bool): Whether to train the model.
            do_eval (bool): Whether to evaluate the model on the validation dataset.
            do_predict (bool): Whether to run predictions on the test dataset.
            train_on_fly (bool, defaults to `True`): Option to return text transformations on the fly (e.g. tokenization
            and augmentation etc.).
        """

        # Set the remove columns parameter based on the dataset inputs
        # We will remove columns based on whether we're running things
        # on the fly or not
        # TODO Change these conditions based on on-the-fly parameters
        if do_predict and not train_on_fly:
            # Remove unused columns from test set - running forward pass
            self._training_args.remove_unused_columns = True
        else:
            # Setting to False so that unused columns can be used for the on the fly transformations
            self._training_args.remove_unused_columns = False

        # Check if train dataset is passed when do_train is True
        if do_train and train_dataset is None:
            raise ValueError("--do_train requires a train dataset")

        # Check if validation dataset is passed when do_eval is True
        # Set the metrics function for validation dataset
        if do_eval:
            if eval_dataset is None:
                raise ValueError("--do_eval requires a validation dataset")

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # Initialize our Trainer
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=MLMMetrics.compute_metrics if do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # Set the number of training and evaluation samples
        self.__train_samples = len(train_dataset) if train_dataset is not None else None
        self.__eval_samples = len(eval_dataset) if eval_dataset is not None else None

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

def main():
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='The file that contains training configurations'
    )
    cli_parser.add_argument(
        '--local_rank',
        type=str,
        help=''
    )
    cli_parser.add_argument(
        '--deepspeed',
        type=str,
        help=''
    )

    args = cli_parser.parse_args()

    # Wan DB project name
    os.environ["WANDB_PROJECT"] = f"icd_embeddings"

    # Huggingface parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # Setup training args, read and store all the other arguments
    model_args, data_args, training_args = parser.parse_json_file(json_file=args.config_file)

    do_train = training_args.do_train
    do_eval = training_args.do_eval
    do_predict = training_args.do_predict
    output_dir = training_args.output_dir
    overwrite_output_dir = training_args.overwrite_output_dir
    resume_from_checkpoint = training_args.resume_from_checkpoint
    seed = training_args.seed

    train_file = data_args.train_file
    validation_file = data_args.validation_file
    test_file = data_args.test_file
    max_train_samples = data_args.max_train_samples
    max_eval_samples = data_args.max_eval_samples
    max_predict_samples = data_args.max_predict_samples
    train_on_fly = data_args.train_on_fly
    validation_on_fly = data_args.validation_on_fly
    test_on_fly = data_args.test_on_fly
    preprocessing_num_workers = data_args.preprocessing_num_workers
    overwrite_cache = data_args.overwrite_cache

    pad_to_max_length = data_args.pad_to_max_length
    max_seq_length = data_args.max_seq_length

    model_name_or_path = model_args.model_name_or_path
    config_name = (
        model_args.config_name if model_args.config_name else model_args.model_name_or_path
    )
    tokenizer_name = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    config_overrides = model_args.config_overrides
    cache_dir = model_args.cache_dir
    model_revision = model_args.model_revision
    use_auth_token = model_args.use_auth_token
    use_pretrained_icd_model = model_args.use_pretrained_icd_model

    text_column_name = data_args.text_column_name
    position_ids_column_name = data_args.position_ids_column_name
    token_type_ids_column_name = data_args.token_type_ids_column_name

    # Load the model helpers object
    model_helpers = ModelHelpers()

    if use_pretrained_icd_model:
        # Load the HuggingFace model config
        config = model_helpers.get_pretrained_config(
            config_name=config_name,
            config_overrides=config_overrides,
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token
        )

        # Load the HuggingFace tokenizer
        tokenizer = model_helpers.get_pretrained_tokenizer(
            tokenizer_name=tokenizer_name,
            model_type=config.model_type,
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token
        )

        # Load the HuggingFace model
        model = model_helpers.get_pretrained_model(
            model_name_or_path=model_name_or_path,
            config=config,
        )
    else:
        # Load the HuggingFace tokenizer
        tokenizer = model_helpers.get_tokenizer(
            tokenizer_path=tokenizer_name,
            name='icd_tokenizer',
            model_max_length=max_seq_length,
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token,
        )

        # Load the HuggingFace model config
        config = model_helpers.get_config(
            tokenizer=tokenizer,
            config_name=config_name,
            config_overrides=config_overrides,
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token
        )

        # Load the HuggingFace model
        model = model_helpers.get_model(
            config=config,
        )

    # Load the M2M PreTrainer object
    mlm_pre_trainer = MLMPreTrainer(
        training_args=training_args,
        config=config,
        tokenizer=tokenizer,
        model=model,
        mlm_probability=0.2,
        fp16=training_args.fp16,
        pad_to_max_length=pad_to_max_length,
        max_seq_length=max_seq_length,
        text_column_name=text_column_name,
        position_ids_column_name=position_ids_column_name,
        token_type_ids_column_name=token_type_ids_column_name,
        seed=seed
    )

    # Load the train dataset
    train_dataset = mlm_pre_trainer.get_train_dataset(
        train_file=train_file,
        training_args=training_args,
        train_on_fly=train_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_train_samples=max_train_samples
    )

    # Load the evaluation dataset - monitor performance on the validation dataset during the course of training
    eval_dataset = mlm_pre_trainer.get_validation_dataset(
        validation_file=validation_file,
        training_args=training_args,
        validation_on_fly=validation_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_eval_samples=max_eval_samples
    )

    test_dataset = mlm_pre_trainer.get_test_dataset(
        test_file=test_file,
        training_args=training_args,
        test_on_fly=test_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_predict_samples=max_predict_samples
    )

    # Set the HuggingFace trainer object based on the training arguments and datasets
    mlm_pre_trainer.set_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        train_on_fly=train_on_fly
    )

    # Detecting last checkpoint.
    last_checkpoint = mlm_pre_trainer.get_checkpoint(
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        do_train=do_train,
        overwrite_output_dir=overwrite_output_dir
    )

    train_metrics = mlm_pre_trainer.run_train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        last_checkpoint=last_checkpoint
    )

    eval_metrics = mlm_pre_trainer.run_eval()

    model_output = mlm_pre_trainer.run_predict(test_dataset=test_dataset)


if __name__ == '__main__':
    main()
