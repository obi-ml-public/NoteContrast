import os
import copy
import sys

import torch
import logging
from pathlib import Path
from gensim.models import Word2Vec
from typing import Union, Optional, NoReturn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from datasets import Dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    BigBirdForSequenceClassification,
    RobertaForSequenceClassification,
    BigBirdForMaskedLM,
    RobertaForMaskedLM
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import ModelArguments, DataTrainingArguments, M2MArguments
from pre_train_datasets.text_tokenizers import SpacyTokenizer
from pre_train_datasets.builder import DatasetBuilder, PreTrainDataset
from pre_train_datasets.data_collator import (
    m2m_data_collator,
    M2MDataCollatorWithPadding,
    M2MDataCollatorForLanguageModeling
)
from pre_train_datasets.text_transformations import (
    M2MTransform,
    TextTransform,
    LabelTransform,
    W2VTransform,
    W2VSmoothedTransform
)

from models.m2m import M2M
from models.m2m.scaling import LogitScale
from models.modeling_m2m import M2MLMForPretrain, M2MForPretrain
from models.m2m.losses import CLIPLoss, CustomMSELoss, CLIPMSELoss, BCELoss, CLIPBCELoss
from models.m2m.losses.weighting import RandomWeighting, UncertaintyWeighting

from training import M2MTrainer
from model_helpers import ModelHelpers
from metrics import M2MMetrics, M2MMultiLabelMetrics

from utils import (
    setup_logging,
    log_training_args
)

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/<path>/phi/ehr_projects/note_pretraining/cache/huggingface/datasets/'


class M2MPreTrainer(object):
    """
    The SequenceTagger class can be used to train sequence tagging models, evaluate models, and
    run predictions. The class can also be used to de-identify and augment text.
    TODO: Add a more detailed description
    """

    def __init__(
            self,
            training_args,
            text_tokenizer: Optional[SpacyTokenizer],
            label_tokenizer: Optional[SpacyTokenizer],
            text_config: PretrainedConfig,
            label_config: Optional[PretrainedConfig],
            text_subword_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            label_subword_tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]],
            text_model: PreTrainedModel,
            label_model: PreTrainedModel,
            text_lm_head,
            text_projection,
            label_projection,
            text_use_projection_head,
            label_use_projection_head,
            gensim_model,
            loss_function: str,
            fp16: bool,
            multi_label,
            lm_objective: Optional[str],
            loss_weighting,
            load_state_dict,
            mlm_probability: float = 0.20,
            smoothing_alpha: Optional[float] = None,
            smoothing_beta: Optional[float] = None,
            smoothing_distance_metric: Optional[str] = None,
            smoothing_reduction: Optional[str] = None,
            text_token_from_end=-1,
            label_token_from_end=-1,
            text_pad_to_max_length: bool = True,
            text_max_seq_length: Optional[int] = None,
            label_pad_to_max_length: bool = True,
            label_max_seq_length: Optional[int] = None,
            text_column_name: str = 'text',
            label_column_name: str = 'label',
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
        self._seed = seed

        # Initialize variables that will be set in the set_trainer function to None
        self._training_args = training_args
        self._trainer = None
        self.__train_samples = None
        self.__eval_samples = None

        # Initialize the tokenizers (HuggingFace and SpaCy), Huggingface config, Huggingface model and data collator
        self._text_tokenizer = text_tokenizer
        self._text_config = text_config
        self._text_subword_tokenizer = text_subword_tokenizer
        self._text_model = text_model

        self._label_tokenizer = label_tokenizer
        self._label_config = label_config
        self._label_subword_tokenizer = label_subword_tokenizer
        self._label_model = label_model

        self._text_lm_head = text_lm_head
        self._gensim_model = gensim_model
        self._text_use_projection_head = text_use_projection_head
        self._label_use_projection_head = label_use_projection_head

        self._lm_objective = lm_objective

        self._multi_label = multi_label
        self._smoothing_alpha = smoothing_alpha
        self._smoothing_beta = smoothing_beta
        self._smoothing_distance_metric = smoothing_distance_metric
        self._smoothing_reduction = smoothing_reduction

        self._data_collator = self.get_data_collator(
            text_pad_to_max_length=text_pad_to_max_length,
            text_max_seq_length=text_max_seq_length,
            label_pad_to_max_length=label_pad_to_max_length,
            label_max_seq_length=label_max_seq_length,
            fp16=fp16,
            text_mlm_probability=mlm_probability,
        )

        # Set the TextTransform
        self._m2m_transform = self.get_m2m_transform(
            text_pad_to_max_length=text_pad_to_max_length,
            text_max_seq_length=text_max_seq_length,
            label_pad_to_max_length=label_pad_to_max_length,
            label_max_seq_length=label_max_seq_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            position_ids_column_name=position_ids_column_name,
            token_type_ids_column_name=token_type_ids_column_name
        )

        self._m2m = self.get_m2m(
            text_projection=text_projection,
            label_projection=label_projection,
            text_token_from_end=text_token_from_end,
            label_token_from_end=label_token_from_end
        )

        self._m2m_loss = self.get_m2m_loss(loss_function=loss_function, num_labels=text_projection)

        self._m2m_model = self.get_m2m_model(loss_weighting=loss_weighting)

        if load_state_dict is not None:
            self._m2m_model.load_state_dict(torch.load(load_state_dict, map_location='cpu'), strict=False)
            if loss_weighting == 'uncertainty':
                # Reset loss weighting
                self._m2m_model.set_loss_weighting(UncertaintyWeighting(num_tasks=2))

            # # TODO: Remove this code
            # if text_max_seq_length > 4096:
            #     self.create_bigbird_distill_model(self._m2m_model._m2m._text_model, text_max_seq_length)

    def get_data_collator(
            self,
            text_pad_to_max_length,
            label_pad_to_max_length,
            text_max_seq_length,
            label_max_seq_length,
            fp16: bool,
            text_mlm_probability,
    ) -> Optional[Union[m2m_data_collator, M2MDataCollatorForLanguageModeling, M2MDataCollatorWithPadding]]:
        """
        Get the data collator object.
        """
        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer,
        # so we change it if we already did the padding.
        # Load data collator
        if text_pad_to_max_length and label_pad_to_max_length and self._text_lm_head is None:
            return m2m_data_collator
        elif self._text_lm_head is not None:
            # Only MLM for now
            return M2MDataCollatorForLanguageModeling(
                text_tokenizer=self._text_subword_tokenizer,
                text_pad_to_multiple_of=8 if fp16 else None,
                text_mlm=self._lm_objective == 'mlm',
                text_mlm_probability=text_mlm_probability,
                label_tokenizer=self._label_subword_tokenizer,
                label_max_length=label_max_seq_length,
                label_pad_to_multiple_of=8 if fp16 else None
            )
        else:
            return M2MDataCollatorWithPadding(
                text_tokenizer=self._text_subword_tokenizer,
                text_pad_to_multiple_of=8 if fp16 else None,
                label_tokenizer=self._label_subword_tokenizer,
                text_padding=True,
                text_max_length=text_max_seq_length,
                label_padding=True,
                label_max_length=label_max_seq_length,
                label_pad_to_multiple_of=8 if fp16 else None
            )

    def get_m2m_transform(
            self,
            text_pad_to_max_length,
            text_max_seq_length,
            label_pad_to_max_length,
            label_max_seq_length,
            text_column_name,
            label_column_name,
            position_ids_column_name,
            token_type_ids_column_name
    ):

        text_transform = TextTransform(
            text_tokenizer=self._text_tokenizer,
            subword_tokenizer=self._text_subword_tokenizer,
            pad_to_max_length=text_pad_to_max_length,
            max_seq_length=text_max_seq_length,
            do_mlm=True if self._text_lm_head is not None else False
        )

        if self._label_subword_tokenizer is not None:
            label_transform = LabelTransform(
                text_tokenizer=self._label_tokenizer,
                subword_tokenizer=self._label_subword_tokenizer,
                pad_to_max_length=label_pad_to_max_length,
                max_seq_length=label_max_seq_length
            )
        else:
            if self._gensim_model is None:
                raise ValueError('Gensim model not specified')
            if self._multi_label:
                label_transform = W2VSmoothedTransform(
                    gensim_model=self._gensim_model,
                    alpha=0.0 if self._smoothing_alpha is None else self._smoothing_alpha,
                    beta=0.0 if self._smoothing_beta is None else self._smoothing_beta,
                    distance_metric=self._smoothing_distance_metric,
                    reduction=self._smoothing_reduction
                )
            else:
                label_transform = W2VTransform(
                    gensim_model=self._gensim_model,
                    text_tokenizer=self._label_tokenizer,
                )

        return M2MTransform(
            text_transform=text_transform,
            label_transform=label_transform,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            position_ids_column_name=position_ids_column_name,
            token_type_ids_column_name=token_type_ids_column_name
        )

    def get_m2m(
            self,
            text_projection,
            label_projection,
            text_token_from_end,
            label_token_from_end
    ):
        # Load the M2M model
        return M2M(
            text_model=self._text_model,
            label_model=self._label_model,
            text_projection=text_projection,
            label_projection=label_projection,
            text_token_from_end=text_token_from_end,
            label_token_from_end=label_token_from_end,
            text_use_projection_head=self._text_use_projection_head,
            label_use_projection_head=self._label_use_projection_head
        )


    def get_m2m_loss(self, loss_function, num_labels):
        logit_scale = LogitScale()
        if loss_function == 'clip':
            # Return the CLIP loss object
            return CLIPLoss(logit_scale=logit_scale)
        elif loss_function == 'mse':
            # Return the MSE loss object
            return CustomMSELoss(num_labels=num_labels)
        elif loss_function == 'clip_mse':
            # Return the CLIP-MSE loss object (Weighted contrastive + mean squared error loss)
            clip_loss = CLIPLoss(logit_scale=logit_scale)
            mse_loss = CustomMSELoss(num_labels=num_labels)
            return CLIPMSELoss(clip_loss=clip_loss, mse_loss=mse_loss)
        elif loss_function == 'bce':
            return BCELoss()
        elif loss_function == 'clip_bce':
            # Return the CLIP-BCE loss object (Weighted contrastive + mean squared error loss)
            clip_loss = CLIPLoss(logit_scale=logit_scale)
            bce_loss = BCELoss()
            return CLIPBCELoss(clip_loss=clip_loss, bce_loss=bce_loss)
        else:
            raise NotImplementedError('Invalid loss function')

    def get_m2m_model(self, loss_weighting):
        # Load the M2M model
        if self._lm_objective in ['clm', 'mlm']:
            if loss_weighting == 'random':
                loss_weighting = RandomWeighting(num_tasks=2)
            elif loss_weighting == 'uncertainty':
                loss_weighting = UncertaintyWeighting(num_tasks=2)
            else:
                raise ValueError('Invalid value for loss weighting')
            return M2MLMForPretrain(
                m2m=self._m2m,
                m2m_loss=self._m2m_loss,
                text_lm_objective=self._lm_objective,
                text_lm_head=self._text_lm_head,
                text_lm_vocab_size=self._text_config.vocab_size,
                loss_weighting=loss_weighting
            )
        else:
            return M2MForPretrain(m2m=self._m2m, m2m_loss=self._m2m_loss)

    def get_train_dataset(
            self,
            train_file: Optional[str],
            training_args: TrainingArguments,
            train_on_fly: bool = True,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            max_train_samples: Optional[int] = None,
            batched: bool = False
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
            batched (bool, defaults to `False`): Process the data in batched mode.

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
            train_dataset.set_transform(self._m2m_transform.encode_train)
        else:
            remove_columns = train_dataset.column_names
            # Run the tokenization process
            with training_args.main_process_first(desc="Train dataset map tokenization"):
                train_dataset = train_dataset.map(
                    self._m2m_transform.encode_train,
                    batched=batched,
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
            max_eval_samples: Optional[int] = None,
            batched: bool = False
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
            batched (bool, defaults to `False`): Process the data in batched mode.

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
            eval_dataset.set_transform(self._m2m_transform.encode_validation)
        else:
            remove_columns = eval_dataset.column_names
            with training_args.main_process_first(desc="Validation dataset map tokenization"):
                eval_dataset = eval_dataset.map(
                    self._m2m_transform.encode_validation,
                    batched=batched,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    remove_columns=remove_columns,
                    desc="Running text transformations on validation dataset",
                )
        return eval_dataset.shuffle(seed=self._seed)

    def get_test_dataset(
            self,
            test_file: Optional[str],
            training_args: TrainingArguments,
            test_on_fly: bool = False,
            preprocessing_num_workers: Optional[int] = None,
            overwrite_cache: bool = False,
            max_predict_samples: Optional[int] = None,
            batched: bool = False
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
            batched (bool, defaults to `False`): Process the data in batched mode.

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
            test_dataset.set_transform(self._m2m_transform.encode_test)
        else:
            with training_args.main_process_first(desc="Test dataset map tokenization"):
                test_dataset = test_dataset.map(
                    self._m2m_transform.encode_test,
                    batched=batched,
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

        # Compute metrics will be None when do_eval is set to False
        compute_metrics = None

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
            if self._multi_label:
                compute_metrics = M2MMultiLabelMetrics(
                    alpha=self._smoothing_alpha,
                    do_lm=self._lm_objective in ['clm', 'mlm'],
                    threshold=0.5
                ).compute
            else:
                compute_metrics = M2MMetrics(do_lm=self._lm_objective in ['clm', 'mlm']).compute

        # Initialize our Trainer
        self._trainer = M2MTrainer(
            model=self._m2m_model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._text_subword_tokenizer,
            data_collator=self._data_collator,
            compute_metrics=compute_metrics,
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

    def save_model(self, return_model_objects=False):

        def get_sequence_model(model, projection, model_type):

            if model_type in ['bigbird', 'big_bird']:
                sequence_model = BigBirdForSequenceClassification(config=model.config)
                sequence_model.bert = model
                sequence_model.classifier = projection
            elif model_type == 'roberta':
                sequence_model = RobertaForSequenceClassification(config=model.config)
                sequence_model.roberta = model
                sequence_model.classifier = projection
            else:
                raise ValueError('Invalid model type')

            return sequence_model

        def get_mlm_model(model, projection, model_type):

            if model_type in ['bigbird', 'big_bird']:
                mlm_model = BigBirdForMaskedLM(config=model.config)
                mlm_model.bert = model
                mlm_model.cls = projection
            elif model_type == 'roberta':
                mlm_model = RobertaForMaskedLM(config=model.config)
                mlm_model.roberta = model
                mlm_model.lm_head = projection
            else:
                raise ValueError('Invalid model type')

            return mlm_model

        training_args = copy.deepcopy(self._training_args)
        output_dir = training_args.output_dir
        self._m2m_model.load_state_dict(torch.load(str(Path(output_dir) / 'pytorch_model.bin')))
        models = {'text_model': [self._m2m_model.get_m2m_text().get_text_model(), self._text_subword_tokenizer]}

        if self._label_config is not None:
            models['label_model'] = [self._m2m_model.get_m2m_text().get_label_model(), self._label_subword_tokenizer]

        if self._text_lm_head is not None:
            models['text_lm_model'] = [
                get_mlm_model(
                    model=models['text_model'][0],
                    projection=self._m2m_model.get_text_lm_head(),
                    model_type='bigbird'
                ),
                self._text_subword_tokenizer
            ]

        if self._m2m_model.get_m2m_text().get_text_head() is not None:

            models['text_sequence_model'] = [
                get_sequence_model(
                    model=models['text_model'][0],
                    projection=self._m2m_model.get_m2m_text().get_text_head(),
                    model_type='bigbird'
                ),
                self._text_subword_tokenizer
            ]

        if self._m2m_model.get_m2m_text().get_label_head() is not None:

            models['label_sequence_model'] = [
                get_sequence_model(
                    model=models['label_model'][0],
                    projection=self._m2m_model.get_m2m_text().get_label_head(),
                    model_type='roberta'
                ),
                self._label_subword_tokenizer
            ]

        for model_name, model_tokenizer in models.items():
            save_path = str(Path(output_dir) / f'{model_name}') + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            training_args.output_dir = save_path
            # Load the trainer
            trainer = M2MTrainer(
                model=model_tokenizer[0],
                args=training_args,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=model_tokenizer[1],
            )
            trainer.save_model()

        if return_model_objects:
            return models

    @staticmethod
    def create_long_model(model, max_pos):
        config = model.config
        current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
        config.max_position_embeddings = max_pos

        assert max_pos > current_max_pos

        # allocate a larger position embedding matrix
        new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
        new_pos_embed[0] = model.embeddings.position_embeddings.weight[0]

        # copy position embeddings over and over to initialize the new position embeddings
        k = 1
        step = current_max_pos - 1

        while k < max_pos - 1:
            new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[1:]
            k += step
        model.embeddings.position_embeddings.weight.data = new_pos_embed
        model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

        return model

    @staticmethod
    def create_bigbird_distill_model(model, max_pos):
        config = model.config
        max_pos += 2
        current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
        config.max_position_embeddings = max_pos

        assert max_pos > current_max_pos

        # allocate a larger position embedding matrix
        new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        step = current_max_pos - 2

        while k < max_pos - 2:
            new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[:-2]
            k += step

        model.embeddings.position_embeddings.weight.data = new_pos_embed
        model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

        return model


def main():
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--text_config_file',
        type=str,
        required=True,
        help='The file that contains training configurations'
    )
    cli_parser.add_argument(
        '--label_config_file',
        type=str,
        required=True,
        help='The file that contains training configurations'
    )
    cli_parser.add_argument(
        '--training_config_file',
        type=str,
        required=True,
        help='The file that contains training configurations'
    )
    cli_parser.add_argument(
        '--m2m_config_file',
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
    cli_parser.add_argument(
        '--save_model',
        action="store_true",
        help='Whether we are running the script to save the model after pre-training'
    )

    args = cli_parser.parse_args()

    # Wan DB project name
    os.environ["WANDB_PROJECT"] = f"note_pretraining_v1"

    # Huggingface parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    training_parser = HfArgumentParser(TrainingArguments)
    m2m_parser = HfArgumentParser(M2MArguments)

    # Setup training args, read and store all the other arguments
    training_args, = training_parser.parse_json_file(json_file=args.training_config_file)
    text_model_args, text_data_args = parser.parse_json_file(json_file=args.text_config_file)
    label_model_args, label_data_args = parser.parse_json_file(json_file=args.label_config_file)
    m2m_args, = m2m_parser.parse_json_file(json_file=args.m2m_config_file)

    do_train = training_args.do_train
    do_eval = training_args.do_eval
    do_predict = training_args.do_predict
    output_dir = training_args.output_dir
    overwrite_output_dir = training_args.overwrite_output_dir
    resume_from_checkpoint = training_args.resume_from_checkpoint
    seed = training_args.seed

    train_file = text_data_args.train_file
    validation_file = text_data_args.validation_file
    test_file = text_data_args.test_file
    max_train_samples = text_data_args.max_train_samples
    max_eval_samples = text_data_args.max_eval_samples
    max_predict_samples = text_data_args.max_predict_samples
    train_on_fly = text_data_args.train_on_fly
    validation_on_fly = text_data_args.validation_on_fly
    test_on_fly = text_data_args.test_on_fly
    preprocessing_num_workers = text_data_args.preprocessing_num_workers
    overwrite_cache = text_data_args.overwrite_cache

    text_pad_to_max_length = text_data_args.pad_to_max_length
    text_max_seq_length = text_data_args.max_seq_length
    label_pad_to_max_length = label_data_args.pad_to_max_length
    label_max_seq_length = label_data_args.max_seq_length

    text_model_name_or_path = text_model_args.model_name_or_path
    text_config_name = (
        text_model_args.config_name if text_model_args.config_name else text_model_args.model_name_or_path
    )
    text_tokenizer_name = (
        text_model_args.tokenizer_name if text_model_args.tokenizer_name else text_model_args.model_name_or_path
    )
    text_config_overrides = text_model_args.config_overrides
    text_spacy_model = text_model_args.spacy_model
    text_cache_dir = text_model_args.cache_dir
    text_model_revision = text_model_args.model_revision
    text_use_auth_token = text_model_args.use_auth_token

    label_model_name_or_path = label_model_args.model_name_or_path
    label_config_name = (
        label_model_args.config_name if label_model_args.config_name else label_model_args.model_name_or_path
    )
    label_tokenizer_name = (
        label_model_args.tokenizer_name if label_model_args.tokenizer_name else label_model_args.model_name_or_path
    )
    label_config_overrides = label_model_args.config_overrides
    label_spacy_model = label_model_args.spacy_model
    label_cache_dir = label_model_args.cache_dir
    label_model_revision = label_model_args.model_revision
    label_use_auth_token = label_model_args.use_auth_token
    use_icd_model = label_model_args.use_icd_model
    use_pretrained_icd_model = label_model_args.use_pretrained_icd_model

    loss_function = m2m_args.loss_function
    lm_objective = m2m_args.lm_objective
    text_projection = m2m_args.text_projection
    label_projection = m2m_args.label_projection
    multi_label = m2m_args.multi_label
    alpha = m2m_args.alpha
    beta = m2m_args.beta
    distance_metric = m2m_args.distance_metric
    reduction = m2m_args.reduction
    loss_weighting = m2m_args.loss_weighting
    load_state_dict = m2m_args.load_state_dict

    text_column_name = (
        text_data_args.text_column_name if text_data_args.text_column_name
        else label_data_args.text_column_name
    )
    label_column_name = (
        text_data_args.label_column_name if text_data_args.label_column_name
        else label_data_args.label_column_name
    )
    position_ids_column_name = (
        label_data_args.position_ids_column_name if label_data_args.position_ids_column_name
        else text_data_args.position_ids_column_name
    )
    token_type_ids_column_name = (
        label_data_args.token_type_ids_column_name if label_data_args.token_type_ids_column_name
        else text_data_args.token_type_ids_column_name
    )
    save_model = args.save_model

    # Load the model helpers object
    model_helpers = ModelHelpers()

    # Load the SpaCy tokenizer object
    if text_spacy_model is None:
        text_tokenizer = None
    else:
        raise NotImplementedError()

    if label_spacy_model is None:
        label_tokenizer = None
    else:
        raise NotImplementedError()

    # Load the HuggingFace model config
    text_config = model_helpers.get_config(
        config_name=text_config_name,
        config_overrides=text_config_overrides,
        cache_dir=text_cache_dir,
        model_revision=text_model_revision,
        use_auth_token=text_use_auth_token
    )
    # Load the HuggingFace model config
    if label_config_name is None:
        label_config = None
    else:
        label_config = model_helpers.get_config(
            config_name=label_config_name,
            config_overrides=label_config_overrides,
            cache_dir=label_cache_dir,
            model_revision=label_model_revision,
            use_auth_token=label_use_auth_token
        )

    # Load the HuggingFace tokenizer
    text_subword_tokenizer = model_helpers.get_tokenizer(
        tokenizer_name=text_tokenizer_name,
        model_type=text_config.model_type,
        cache_dir=text_cache_dir,
        model_revision=text_model_revision,
        use_auth_token=text_use_auth_token
    )

    if label_tokenizer_name is None:
        label_subword_tokenizer = None
    else:
        label_subword_tokenizer = model_helpers.get_tokenizer(
            tokenizer_name=label_tokenizer_name,
            model_type=label_config.model_type,
            cache_dir=label_cache_dir,
            model_revision=label_model_revision,
            use_auth_token=label_use_auth_token
        )

    if use_icd_model and label_config_name is not None:
        label_config = model_helpers.get_icd_config(
            tokenizer=label_subword_tokenizer,
            config_name=label_config_name,
            config_overrides=label_config_overrides,
            cache_dir=label_cache_dir,
            model_revision=label_model_revision,
            use_auth_token=label_use_auth_token
        )

    # Load the HuggingFace model
    text_model = model_helpers.get_text_model(
        model_name_or_path=text_model_name_or_path,
        config=text_config,
        from_tf=bool(".ckpt" in text_model_name_or_path),
        cache_dir=text_cache_dir,
        model_revision=text_model_revision,
        use_auth_token=text_use_auth_token
    )
    if lm_objective is not None:
        text_lm_head = model_helpers.get_mlm_head(
            model_name_or_path=text_model_name_or_path,
            config=text_config,
            cache_dir=text_cache_dir,
            model_revision=text_model_revision,
            use_auth_token=text_use_auth_token,
        )
    else:
        text_lm_head = None

    if label_config is None:
        label_model = model_helpers.get_labeled_model()
    else:
        if use_pretrained_icd_model:
            label_model = model_helpers.get_icd_pretrained_model(label_model_name_or_path, config=label_config)
        else:
            label_model = model_helpers.get_icd_model(config=label_config)

    if m2m_args.gensim_model is not None:
        gensim_model = Word2Vec.load(m2m_args.gensim_model)
    else:
        gensim_model = None

    training_args.label_names = ['m2m_texts', 'm2m_labels']

    if save_model:
        load_state_dict = None

    # Load the M2M PreTrainer object
    m2m_pre_trainer = M2MPreTrainer(
        training_args=training_args,
        text_tokenizer=text_tokenizer,
        label_tokenizer=label_tokenizer,
        text_config=text_config,
        label_config=label_config,
        text_subword_tokenizer=text_subword_tokenizer,
        label_subword_tokenizer=label_subword_tokenizer,
        text_model=text_model,
        label_model=label_model,
        text_lm_head=text_lm_head,
        text_projection=text_projection,
        label_projection=label_projection,
        text_use_projection_head=not multi_label,
        label_use_projection_head=not multi_label,
        gensim_model=gensim_model,
        loss_function=loss_function,
        multi_label=multi_label,
        fp16=training_args.fp16,
        lm_objective=lm_objective,
        loss_weighting=loss_weighting,
        load_state_dict=load_state_dict,
        smoothing_alpha=alpha,
        smoothing_beta=beta,
        smoothing_reduction=reduction,
        smoothing_distance_metric=distance_metric,
        text_pad_to_max_length=text_pad_to_max_length,
        text_max_seq_length=text_max_seq_length,
        label_pad_to_max_length=label_pad_to_max_length,
        label_max_seq_length=label_max_seq_length,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        position_ids_column_name=position_ids_column_name,
        token_type_ids_column_name=token_type_ids_column_name,
        seed=seed
    )

    # if save_model:
    #     m2m_pre_trainer.save_model()
    #     sys.exit()

    # Load the train dataset
    train_dataset = m2m_pre_trainer.get_train_dataset(
        train_file=train_file,
        training_args=training_args,
        train_on_fly=train_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_train_samples=max_train_samples
    )

    # Load the evaluation dataset - monitor performance on the validation dataset during the course of training
    eval_dataset = m2m_pre_trainer.get_validation_dataset(
        validation_file=validation_file,
        training_args=training_args,
        validation_on_fly=validation_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_eval_samples=max_eval_samples
    )

    test_dataset = m2m_pre_trainer.get_test_dataset(
        test_file=test_file,
        training_args=training_args,
        test_on_fly=test_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_predict_samples=max_predict_samples
    )

    # Set the HuggingFace trainer object based on the training arguments and datasets
    m2m_pre_trainer.set_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        train_on_fly=train_on_fly
    )

    # Detecting last checkpoint.
    last_checkpoint = m2m_pre_trainer.get_checkpoint(
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        do_train=do_train,
        overwrite_output_dir=overwrite_output_dir
    )

    train_metrics = m2m_pre_trainer.run_train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        last_checkpoint=last_checkpoint
    )

    eval_metrics = m2m_pre_trainer.run_eval()

    model_output = m2m_pre_trainer.run_predict(test_dataset=test_dataset)


if __name__ == '__main__':
    main()
