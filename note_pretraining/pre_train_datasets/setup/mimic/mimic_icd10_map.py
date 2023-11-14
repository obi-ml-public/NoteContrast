import re
import json
import icd10
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer

mapping_file = './data_files/icd_9_to_icd10_mapping.csv'

mapping_df = pd.read_csv(mapping_file, sep='|')

tokenizer_name = '/mnt/<path>/phi/ehr_projects/icd_embeddings/models/v0/no_sampling_trial_2/checkpoint-200000/'

vocab = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True).vocab

codes_map_dict = defaultdict(list)
for row in mapping_df.itertuples():

    icd_10_code = row.TargetI9
    icd_9_code = row.Flags

    if vocab.get(icd_10_code, None) is not None or icd10.exists(icd_10_code):
        codes_map_dict[icd_9_code].append(icd_10_code)

for key, value in codes_map_dict.items():
    if len(value) > 1:
        precise = [l for l in value if '.' in l]
        if len(precise) > 0:
            value = precise
            codes_map_dict[key] = value

codes_map_dict = {k: v for k, v in codes_map_dict.items()}


def clean_text(text, long_char_max=8):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)
    text = text.replace('\n', ' ').replace("\r", " ").strip()
    return re.sub(
        rf'(?P<specchar>([^a-zA-Z0-9_\s]|_)){{{long_char_max},}}', r'\g<specchar>' * long_char_max, text
    )


split = 'train'
dataset = 'full'

input_file = \
    '/mnt/<path>/phi/ehr_projects/phenotype_classification/data/multi_label_mimic_icd_9/' + \
    f'{dataset}/{split}.jsonl'

output_file = f'/mnt/obi0/phi/ehr_projects/note_pretraining/data/mimic_3/{split}.jsonl'
c = 0

with open(output_file, 'w') as file:
    for line in open(input_file):
        # Load the data
        data = json.loads(line)
        # Split the labels
        labels = data.pop('labels').split()
        # Map ICD-9 to ICD-10 labels
        icd10_labels = [
            np.random.choice(codes_map_dict[label])
            for label in labels if codes_map_dict.get(label, None) is not None
        ]
        if not icd10_labels:
            c += 1
            print(c)
            continue

        # Create ID's
        data['patient_id'] = data['subject_id']
        data['patient_encounter_id'] = data.pop('subject_id')
        data['note_id'] = data.pop('hadm_id')
        # Clean the text
        data['text'] = clean_text(data['text'])
        # Create the labels
        data['label_patient'] = ' '.join(sorted(icd10_labels))
        data['label_note'] = data['label_patient']
        data['position_ids'] = ('0 ' * len(icd10_labels)).strip()
        data['token_type_ids'] = ('1 ' * len(icd10_labels)).strip()
        data['position_ids_note'] = data['position_ids']
        data['token_type_ids_note'] = data['token_type_ids']

        file.write(json.dumps(data) + '\n')