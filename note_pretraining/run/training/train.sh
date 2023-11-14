# MLM Objective

deepspeed /mnt/<path>/<user>/projects/note_pretraining/mlm_pre_trainer.py \
--config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_mimic_3_v2/training.json \
--deepspeed /mnt/<path>/<user>/projects/note_pretraining/deepspeed_config.json

# Contrastive Training

deepspeed /mnt/<path>/<user>/projects/note_pretraining/m2m_pre_trainer.py \
--text_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1/text.json \
--label_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1/label.json \
--training_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1/training.json \
--m2m_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1/m2m.json \
--deepspeed /mnt/<path>/<user>/projects/note_pretraining/deepspeed_config.json

deepspeed /mnt/<path>/<user>/projects/note_pretraining/m2m_pre_trainer.py \
--text_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192/text.json \
--label_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192/label.json \
--training_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192/training.json \
--m2m_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192/m2m.json \
--deepspeed /mnt/<path>/<user>/projects/note_pretraining/deepspeed_config.json

deepspeed /mnt/<path>/<user>/projects/note_pretraining/m2m_pre_trainer.py \
--text_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd/text.json \
--label_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd/label.json \
--training_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd/training.json \
--m2m_config_file /mnt/<path>/<user>/projects/note_pretraining/config_files/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd/m2m.json \
--deepspeed /mnt/<path>/<user>/projects/note_pretraining/deepspeed_config.json