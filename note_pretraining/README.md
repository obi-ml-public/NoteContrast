# Medical Note Pre-training

**Pre-training medical text models via contrastive learning and language modeling.**

*Version 1.0.0 / 20 June 2022*

*We're working on expanding the documentation and readme. While we work on that, taking a look at the source code may help answer some questions.*

*Comments, feedback and improvements are welcome and encouraged!*

---

# Project Details 

*This section provides information about the project. This includes information about installation, project features, and a general project overview.*
*It is useful for getting a brief understanding of the project and submitting any questions to the project team*

<details>
<summary>Click to expand</summary>

* This repository was used to pre-train medical text models using masked language modeling, contrastive learning or both.
* We trained different models for medical text which support document lengths of up to 8192 tokens. We used the BioLM RoBERTa-base-PM-M3-Voc-distill-align as the starting model checkpoint.

</details>

---

# Dataset Creation
*This section provides information on how the dataset of medical text was created*

<details>
<summary>Click to expand</summary>

* The NoteLM text model was trained on medical notes from the MIMIC-III dataset, a collection of medical notes from over 40,000 patients. 
* We used nearly 2 million 356 notes without available diagnostic codes for the masked language model pre-training and a set of nearly 50,000 discharge summaries with available diagnostic codes for the contrastive language-diagnostic pre-training step.
* We also created a dataset of textual descriptions of ICD-10 codes.
* The code to pre-process the data, remove PHI placeholders, obtain textual descriptions and map ICD-9 to ICD-10 codes can be found in: [pre_train_datasets/setup/mimic](pre_train_datasets/setup/mimic)

</details>

---

# Training
*This section provides information about the training data, the speed and size of training elements.*
*It is useful for people who want to learn more about the model inputs, objective, architecture, development, training/evaluation/prediction procedure and compute infrastructure.*

<details>
<summary>Click to expand</summary>

## MLM Training

* We train one model (NoteLM 4k) with a maximum context length of 4096 tokens on 2 million medical notes from the mimic dataset.
* We use the masked language modeling objective with a 20% mask percentage.
* The code to train the model is located here: [mlm_pre_trainer.py](mlm_pre_trainer.py)
* The configurations used to train the model is here: [config_files_emlnp/train/mlm_mimic_3_v2](config_files_emlnp/train/mlm_mimic_3_v2)

## Contrastive Training

* We train 4 models with the contrastive loss and the MLM objective. Two of those models (NoteContrast 8k & NoteContrast 8k ICD) can handle upto 8192 tokens, while the other handles 4096 (NoteContrast 4k) tokens.
* The second model of length 8192 is the model fine-tuned with the ICD-10 descriptions using the contrastive loss.
* The code for building the contrastive learning architecture can be found in this folder: [models](models)
* The code to train the models is located here: [m2m_pre_trainer.py](m2m_pre_trainer.py)
* The configurations used to train the NoteContrast 4k model: [config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1](config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1), the NoteContrast 8k: [config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1_8192](config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1_8192) and the NoteContrast 8k ICD: [config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd](config_files_emlnp/train/mlm_icd10_clip_note_mimic_3_v1_8192_icd)

We used the development set to select the best performing model during the course of training.
We used deepspeed while training our model, and the commands for training all the models are present in: [run_emnlp/training/train.sh](run_emnlp/training/train.sh)

</details>

---