import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from transformers import (
    HfArgumentParser
)

from model_helpers import ModelHelpers
from arguments import ModelArguments

class ICDProjections(object):

    def __init__(
            self,
            tokenizer,
            model,
            icd_column='icd10cm',
            embedding_column='embedding'
    ):
        self._model = model
        self._vocabulary = tokenizer.vocab
        self._icd_column = icd_column
        self._embedding_column = embedding_column
        self._embeddings_df = self.get_embeddings()

    def get_embeddings(self):
        return pd.DataFrame(
            {
                self._icd_column: sorted(self._vocabulary, key=lambda k: self._vocabulary[k]),
                self._embedding_column: [
                    weight.tolist() for weight in self._model.get_input_embeddings().weight.detach().numpy()
                ]
            }
        )

config_file = './config_files/predict/projections.json'

# Huggingface parser
parser = HfArgumentParser(ModelArguments)

# Setup training args, read and store all the other arguments
model_args,  = parser.parse_json_file(json_file=config_file)

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


# Load the model helpers object
model_helpers = ModelHelpers()

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

icd_projections = ICDProjections(
    tokenizer=tokenizer,
    model=model,
    icd_column='icd10cm',
    embedding_column='embedding'
)

embedding_df = icd_projections._embeddings_df[
    ~icd_projections._embeddings_df['icd10cm'].isin(tokenizer.special_tokens_map.values())
]

embedding_df.reset_index(drop=True, inplace=True)

output_file = '/mnt/obi0/phi/ehr_projects/icd_embeddings/embeddings/projections.parquet'

embedding_df.to_parquet(output_file)