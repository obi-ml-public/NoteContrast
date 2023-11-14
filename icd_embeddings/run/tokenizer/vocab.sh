python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_vocab.py \
--input_file /mnt/<path>/phi/ehr_projects/ehr_active/data/icd10_pretraining/ecg_hem_vir/encounters_6mo_bins_long.parquet \
--cutoff 10 \
--vocab_file /mnt/<path>/phi/ehr_projects/icd_embeddings/data/vocab.txt

python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_vocab.py \
--input_file /mnt/<path>/phi/ehr_projects/ehr_active/data/icd10_pretraining/ecg_hem_vir/encounters_6mo_bins_long.parquet \
--cutoff 0 \
--vocab_file /mnt/<path>/phi/ehr_projects/icd_embeddings/data/vocab.txt

python /mnt/<path>/<user>/projects/icd_embeddings/icd_tokenizers/create_vocab.py \
--input_file /mnt/<path>/phi/ehr_projects/ehr_active/data/icd10_pretraining/ecg_hem_vir/encounters_6mo_bins_long.parquet \
--cutoff 0 \
--vocab_file /mnt/<path>/phi/ehr_projects/sequence_embeddings/data/bloodcell_clip/vocabulary/vocab_ecg_hem_vir.txt