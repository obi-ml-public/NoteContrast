import os
import json
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments
from model_helpers import ModelHelpers
from models.heads import ClassificationHead
from text_classifier import TextClassifier
from prompt_text_classifier import PromptTextClassifier
from mimic_full_classifier import MimicFullClassifier


logger = logging.getLogger(__name__)
HF_DATASETS_CACHE = '/mnt/obi0/phi/ehr_projects/phenotype_classification/cache/huggingface/datasets/'


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
    cli_parser.add_argument(
        '--eval_output_file',
        type=str,
        default=None,
        help=''
    )

    args = cli_parser.parse_args()

    # Huggingface parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=args.config_file)

    # Setup training args, read and store all the other arguments
    do_train = training_args.do_train
    do_eval = training_args.do_eval
    do_predict = training_args.do_predict
    output_dir = training_args.output_dir
    overwrite_output_dir = training_args.overwrite_output_dir
    resume_from_checkpoint = training_args.resume_from_checkpoint
    seed = training_args.seed

    label_list = data_args.label_list
    label_list.sort()
    prompt_label_list = data_args.prompt_label_list
    ignore_labels = data_args.ignore_labels
    # ignore_labels.sort()
    if prompt_label_list is not None:
        prompt_label_list.sort()

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
    spacy_model = model_args.spacy_model
    cache_dir = model_args.cache_dir
    model_revision = model_args.model_revision
    use_auth_token = model_args.use_auth_token

    problem_type = model_args.problem_type
    use_probing_classifier = model_args.use_probing_classifier
    use_prompting_classifier = model_args.use_prompting_classifier
    classification_threshold = model_args.classification_threshold
    optimal_f1_threshold = model_args.optimal_f1_threshold

    text_column_name = data_args.text_column_name
    label_column_name = data_args.label_column_name

    set_seed(seed)

    if use_prompting_classifier:
        # Wan DB project name
        os.environ["WANDB_PROJECT"] = f"prompt_icd"
    else:
        os.environ["WANDB_PROJECT"] = f"downstream_classification"

    # Load the model helpers object
    model_helpers = ModelHelpers()

    # Load the SpaCy tokenizer object
    if spacy_model is not None:
        text_tokenizer = model_helpers.get_text_tokenizer(spacy_model=spacy_model)
    else:
        text_tokenizer = None

    # Load the HuggingFace model config
    config = model_helpers.get_config(
        label_list=label_list,
        problem_type=problem_type,
        config_name=config_name,
        config_overrides=config_overrides,
        cache_dir=cache_dir,
        model_revision=model_revision,
        use_auth_token=use_auth_token
    )

    # Load the HuggingFace tokenizer
    tokenizer = model_helpers.get_tokenizer(
        tokenizer_name=tokenizer_name,
        model_type=config.model_type,
        cache_dir=cache_dir,
        model_revision=model_revision,
        use_auth_token=use_auth_token
    )

    if use_prompting_classifier:
        # Load the HuggingFace model
        model = model_helpers.get_prompting_model(
            model_name_or_path=model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_name_or_path),
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token
        )
    else:
        # Load the HuggingFace model
        model = model_helpers.get_model(
            model_name_or_path=model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_name_or_path),
            cache_dir=cache_dir,
            model_revision=model_revision,
            use_auth_token=use_auth_token
        )

    if use_probing_classifier:
        for param in model.base_model.parameters():
            param.requires_grad = False
        classification_head = ClassificationHead(config=config)
        classification_head = classification_head.apply(classification_head.init_weights)
        model.classifier = classification_head

    if use_prompting_classifier:
        mimic_full = False
        if mimic_full:
            codes = set()
            for split in ['train', 'dev', 'test']:
                input_file = f'/mnt/obi0/phi/ehr/mimic-3/physionet.org/files/mimiciii/1.4/mimic3_{split}.json'
                with open(input_file, 'r') as file:
                    notes = json.load(file)
                for note in notes:
                    for label in note['LABELS'].split(';'):
                        if label.endswith('.'):
                            label = label[:-1]
                        codes.add(label)
            # Load the sequence tagger object
            text_classifier = MimicFullClassifier(
                text_tokenizer=text_tokenizer,
                config=config,
                subword_tokenizer=tokenizer,
                model=model,
                label_list=label_list,
                ignore_labels=ignore_labels,
                validation_file=validation_file,
                icd_codes=codes,
                fp16=training_args.fp16,
                pad_to_max_length=pad_to_max_length,
                max_seq_length=max_seq_length,
                text_column_name=text_column_name,
                label_column_name=label_column_name,
                seed=seed,
                classification_threshold=classification_threshold,
                optimal_f1_threshold=optimal_f1_threshold
            )
        else:
            # Load the sequence tagger object
            text_classifier = PromptTextClassifier(
                text_tokenizer=text_tokenizer,
                config=config,
                subword_tokenizer=tokenizer,
                model=model,
                label_list=label_list,
                ignore_labels=ignore_labels,
                fp16=training_args.fp16,
                prompt_label_list=prompt_label_list,
                pad_to_max_length=pad_to_max_length,
                max_seq_length=max_seq_length,
                text_column_name=text_column_name,
                label_column_name=label_column_name,
                seed=seed,
                classification_threshold=classification_threshold,
                optimal_f1_threshold=optimal_f1_threshold
            )
    else:
        # Load the sequence tagger object
        text_classifier = TextClassifier(
            text_tokenizer=text_tokenizer,
            config=config,
            subword_tokenizer=tokenizer,
            model=model,
            label_list=label_list,
            ignore_labels=ignore_labels,
            fp16=training_args.fp16,
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            seed=seed,
            classification_threshold=classification_threshold,
            optimal_f1_threshold=optimal_f1_threshold
        )

    # Load the train dataset
    train_dataset = text_classifier.get_train_dataset(
        train_file=train_file,
        training_args=training_args,
        train_on_fly=train_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_train_samples=max_train_samples
    )

    # Load the evaluation dataset - monitor performance on the validation dataset during the course of training
    eval_dataset = text_classifier.get_validation_dataset(
        validation_file=validation_file,
        training_args=training_args,
        validation_on_fly=validation_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_eval_samples=max_eval_samples
    )

    test_dataset = text_classifier.get_test_dataset(
        test_file=test_file,
        training_args=training_args,
        test_on_fly=test_on_fly,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        max_predict_samples=max_predict_samples
    )

    # Set the HuggingFace trainer object based on the training arguments and datasets
    text_classifier.set_trainer(
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict
    )

    # Detecting last checkpoint.
    last_checkpoint = text_classifier.get_checkpoint(
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        do_train=do_train,
        overwrite_output_dir=overwrite_output_dir
    )

    train_metrics = text_classifier.run_train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        last_checkpoint=last_checkpoint
    )

    eval_metrics = text_classifier.run_eval()

    model_output = text_classifier.run_predict(test_dataset=test_dataset)

    eval_output_file = args.eval_output_file
    if eval_output_file is not None:
        from pathlib import Path
        Path(eval_output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_output_file, 'w') as file:
            json.dump(eval_metrics, file)

if __name__ == '__main__':
    main()
