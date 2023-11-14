import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

input_file = '/mnt/<path>/phi/ehr/mimic-3/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv'

train_file = '/mnt/<path>/phi/ehr_projects/note_pretraining/data/mimic_3/train_mlm.jsonl'
validation_file = '/mnt/<path>/phi/ehr_projects/note_pretraining/data/mimic_3/validation_mlm.jsonl'

mimic_df = pd.read_csv(input_file)

note_text_column = 'TEXT'

def clean_text(text, long_char_max=8):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)
    text = text.replace('\n', ' ').replace("\r", " ").strip()
    text = re.sub('\s+', ' ', text)
    return re.sub(
        rf'(?P<specchar>([^a-zA-Z0-9_\s]|_)){{{long_char_max},}}', r'\g<specchar>' * long_char_max, text
    )

mimic_df['text'] = mimic_df['TEXT'].apply(clean_text)

train_subjects, test_subjects = train_test_split(mimic_df.SUBJECT_ID.unique(), test_size=0.01)

train_df = mimic_df[mimic_df['SUBJECT_ID'].isin(train_subjects)]
validation_df = mimic_df[mimic_df['SUBJECT_ID'].isin(test_subjects)]

print(train_df.shape)

print(validation_df.shape)

train_df.ROW_ID.nunique()

with open(train_file, 'w') as file:
    for row in train_df.itertuples():
        if row.text.strip() == '' or row.text is None:
            continue
        json_obj = {'note_id': str(int(row.ROW_ID)), 'patient_id': str(row.SUBJECT_ID), 'text': row.text}
        file.write(json.dumps(json_obj) + '\n')

with open(validation_file, 'w') as file:
    for row in validation_df.itertuples():
        if row.text.strip() == '' or row.text is None:
            continue
        json_obj = {'note_id': str(int(row.ROW_ID)), 'patient_id': str(row.SUBJECT_ID), 'text': row.text}
        file.write(json.dumps(json_obj) + '\n')