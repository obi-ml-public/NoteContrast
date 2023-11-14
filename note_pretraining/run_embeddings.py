import json
from pathlib import Path

import torch
from argparse import ArgumentParser

from transformers import (
    HfArgumentParser,
    TrainingArguments, BigBirdForSequenceClassification,
)

from model_helpers import ModelHelpers
from mlm_pre_trainer import MLMPreTrainer
from arguments import ModelArguments, DataTrainingArguments
from models.icd.modeling_icd_roberta import ICDRobertaForSequenceClassification
from pre_train_datasets.data_collator.data_collator import ICDDataCollator

from pre_train_datasets.text_transformations import (
    MLMTransform,
    TextTransform,
    LabelTransform,
    ICDMLMTransform
)

from transformers import DataCollatorWithPadding

cli_parser = ArgumentParser(
    description='configuration arguments provided at run time from the CLI',
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

# Huggingface parser
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

config_file = f'/mnt/obi0/pk621/projects/note_pretraining/config_files/train/embeddings/label.json'

# Setup training args, read and store all the other arguments
model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)

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
spacy_model = model_args.spacy_model
cache_dir = model_args.cache_dir
model_revision = model_args.model_revision
use_auth_token = model_args.use_auth_token

text_column_name = data_args.text_column_name

icd = True

# Load the model helpers object
model_helpers = ModelHelpers()

# Load the SpaCy tokenizer object
if spacy_model is None:
    text_tokenizer = None
else:
    raise NotImplementedError()

# Load the HuggingFace model config
config = model_helpers.get_config(
    config_name=config_name,
    config_overrides=config_overrides,
    cache_dir=cache_dir,
    model_revision=model_revision,
    use_auth_token=use_auth_token
)
config.output_hidden_states = True

# Load the HuggingFace tokenizer
subword_tokenizer = model_helpers.get_tokenizer(
    tokenizer_name=tokenizer_name,
    model_type=config.model_type,
    cache_dir=cache_dir,
    model_revision=model_revision,
    use_auth_token=use_auth_token
)

def get_icd_sequence_model(
        model_name_or_path,
        config,
):
    """
    Get the HuggingFace model

    Args:
        model_name_or_path (str): The model checkpoint for weights initialization.
        config (PretrainedConfig): The HuggingFace config object

    Returns:
        (ICDRobertaModel): The HuggingFace ICD Roberta model object
    """

    return ICDRobertaForSequenceClassification.from_pretrained(model_name_or_path, config=config)

if icd:
    from models.m2m.heads import ProjectionHead
    model = get_icd_sequence_model(
        model_name_or_path=model_name_or_path,
        config=config,
    )
    model.classifier = ProjectionHead(config=config, projection=768, token_from_end=-1)
    model.load_state_dict(torch.load(str(Path(model_name_or_path) / 'pytorch_model.bin')), strict=False)
else:
    from models.m2m.heads import ProjectionHead
    # Load the HuggingFace model
    model = BigBirdForSequenceClassification(config=config)
    model.classifier = ProjectionHead(config=config, projection=768, token_from_end=-1)
    model.load_state_dict(torch.load(str(Path(model_name_or_path) / 'pytorch_model.bin')), strict=False)

# Load the M2M PreTrainer object
mlm_pre_trainer = MLMPreTrainer(
    training_args=training_args,
    tokenizer=text_tokenizer,
    config=config,
    subword_tokenizer=subword_tokenizer,
    model=model,
    mlm_probability=0.2,
    fp16=training_args.fp16,
    pad_to_max_length=pad_to_max_length,
    max_seq_length=max_seq_length,
    text_column_name=text_column_name,
    seed=seed
)

if icd:
    data_collator = DataCollatorWithPadding(
        tokenizer=subword_tokenizer,
        padding=True,
        max_length=max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        return_tensors='pt'
    )
    # Reset data collator
    mlm_pre_trainer._data_collator = ICDDataCollator(
        tokenizer=subword_tokenizer,
        data_collator=data_collator,
    )
else:
    # Reset data collator
    mlm_pre_trainer._data_collator = DataCollatorWithPadding(
        tokenizer=subword_tokenizer,
        padding=True,
        max_length=max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        return_tensors='pt'
    )

if icd:
    label_transform = LabelTransform(
        text_tokenizer=text_tokenizer,
        subword_tokenizer=subword_tokenizer,
        pad_to_max_length=pad_to_max_length,
        max_seq_length=max_seq_length
    )
    mlm_pre_trainer._mlm_transform = ICDMLMTransform(
        transform=label_transform,
        text_column_name=text_column_name,
        position_ids_column_name='position_ids_note',
        token_type_ids_column_name='token_type_ids_note'
    )
else:
    text_transform = TextTransform(
        text_tokenizer=text_tokenizer,
        subword_tokenizer=subword_tokenizer,
        pad_to_max_length=pad_to_max_length,
        max_seq_length=max_seq_length,
        do_mlm=False
    )
    mlm_pre_trainer._mlm_transform = MLMTransform(
        text_transform=text_transform,
        text_column_name=text_column_name,
    )

test_dataset = mlm_pre_trainer.get_test_dataset(
    test_file=test_file,
    training_args=training_args,
    test_on_fly=test_on_fly,
    preprocessing_num_workers=preprocessing_num_workers,
    overwrite_cache=overwrite_cache,
    max_predict_samples=max_predict_samples
)
# test_dataset = test_dataset.remove_columns('labels')

# Set the HuggingFace trainer object based on the training arguments and datasets
mlm_pre_trainer.set_trainer(
    train_dataset=None,
    eval_dataset=None,
    do_train=do_train,
    do_eval=do_eval,
    do_predict=do_predict,
    train_on_fly=False
)

model_output = mlm_pre_trainer.run_predict(test_dataset=test_dataset)

output_file = '/mnt/obi0/phi/ehr_projects/note_pretraining/embeddings/label/checkpoint_final.jsonl'

vocab = subword_tokenizer.vocab
input_embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

with open(output_file, 'w') as file:
    for test_data, clip_projection, hidden_state_cls, hidden_state in zip(
            test_dataset,
            model_output.predictions['clip_projection'],
            model_output.predictions['hidden_state_cls'],
            model_output.predictions['hidden_state']
    ):
        json_obj = {
            'note_id': test_data['note_id'],
            'text': test_data['note_id'],
            'clip_projection':clip_projection.tolist(),
            'hidden_state_cls': hidden_state_cls.tolist(),
            'hidden_state_code': hidden_state.tolist(),
            'input_embedding': input_embeddings[vocab[test_data['note_id']]].tolist()
        }
        file.write(json.dumps(json_obj) + '\n')

# with open(output_file, 'w') as file:
#     for test_data, prediction in zip(test_dataset, model_output.predictions):
#         json_obj = {
#             'note_id': test_data['note_id'],
#             'text': test_data['text'],
#             'embedding':prediction.tolist()
#         }
#         file.write(json.dumps(json_obj) + '\n')