python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_tokenizer.py \
--files /mnt/<path>/phi/ehr_projects/icd_embeddings/data/vocab.txt \
--save_file /mnt/<path>/phi/ehr_projects/icd_embeddings/tokenizers/icd10_roberta.json

python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_tokenizer.py \
--files /mnt/<path>/phi/ehr_projects/icd_embeddings/data/vocab.txt \
--save_file /mnt/<path>/phi/ehr_projects/icd_embeddings/tokenizers/icd10_deberta.json \
--special_tokens \[CLS\] \[SEP\] \[UNK\] \[SEP\] \[PAD\] \[CLS\] \[MASK\]

python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_tokenizer.py \
--files /mnt/<path>/phi/ehr_projects/icd_embeddings/data/vocab.txt \
--save_file /mnt/<path>/phi/ehr_projects/icd_embeddings/tokenizers/icd10_deberta.json \
--special_tokens [CLS] [SEP] [UNK] [PAD] [MASK]