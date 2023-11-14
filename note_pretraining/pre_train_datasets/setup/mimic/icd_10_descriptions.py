import re
import json
import icd10
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer

tokenizer_name = '/mnt/<path>/phi/ehr_projects/icd_embeddings/models/v0/no_sampling_trial_2/checkpoint-200000/'

output_file = f'/mnt/<path>/phi/ehr_projects/note_pretraining/data/icd_10_descriptions/train.jsonl'

vocab = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True).vocab

with open(output_file, 'w') as file:
    for icd_code in vocab.keys():

        if icd_code in ['<mask>', '<unk>', '<s>', '</s>', '<pad>']:
            continue

        icd_object = icd10.find(icd_code)

        description = icd_object.description

        if not re.search('\w+', description):
            raise ValueError()

        data = {
            'patient_id': icd_code,
            'patient_encounter_id': icd_code,
            'note_id': icd_code,
            'text': description.replace('\n', ' ').replace("\r", " ").strip(),
            'label_patient': icd_code,
            'label_note': icd_code,
            'position_ids': '0',
            'token_type_ids': '1',
            'position_ids_note': '0',
            'token_type_ids_note': '1'
        }

        file.write(json.dumps(data) + '\n')