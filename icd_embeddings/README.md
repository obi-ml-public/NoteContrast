# ICD Embeddings

**Transformer based embedding models for diagnostic codes.**

*Version 1.0.0 / 20 June 2022*

*We're working on expanding the documentation and readme. While we work on that, taking a look at the source code may help answer some questions.*

*Comments, feedback and improvements are welcome and encouraged!*

---

# Project Details 

*This section provides information about the project. This includes information about installation, project features, and a general project overview.*
*It is useful for getting a brief understanding of the project and submitting any questions to the project team*

<details>
<summary>Click to expand</summary>

* This repository was used to train contextual embedding models for  ICD-10 sequences diagnostic codes based on a large real-world data set.
* We used the RoBERTa architecture to model sequences of diagnostic codes.

</details>

---

# Dataset Creation
*This section provides information on how the dataset of ICD-10 sequences were created*

<details>
<summary>Click to expand</summary>

* To learn long-term temporal associations, we represented sequences of ICD-10 codes across multiple clinical encounters of a patient over time. 
* We selected one encounter as the “encounter of interest”, and calculated the time difference (in days) for past and future encounters, with 0 indicating all diagnostic codes in the current encounter.
* We used nearly 60 million real-world hospital encounters from 1.5 million patients. We first prepared a sequence for each patient containing all available ICD-10 diagnostic codes, leading to 1.5 million sequences of varying lengths. 
* We then randomly selected an encounter to be the “current encounter” and by changing the current encounter, generated 5 sequences per patient that contained the same sequence of codes, but different 349 relative position and token type values, resulting in a final dataset of 7.5 million sequences.
* The code to process the data can be found in: [pre_train_datasets/setup/icd_dataset.py](pre_train_datasets/setup/icd_dataset.py)
* We run the script using the command present in: [run_emnlp/dataset/icd_dataset_partition/0.sh](run_emnlp/dataset/icd_dataset_partition/0.sh)

</details>

---

# Training
*This section provides information about the training data, the speed and size of training elements.*
*It is useful for people who want to learn more about the model inputs, objective, architecture, development, training/evaluation/prediction procedure and compute infrastructure.*

<details>
<summary>Click to expand</summary>

* We created a tokenizer for ICD-10 codes. The tokenizer is a simple whitespace tokenizer. The code for building the ICD-10 vocabulary and creating the tokenizer is present in the following folder: [icd_tokenizers](icd_tokenizers)
* We trained the ICD-10 sequence model using the masked language modeling objective, where 20% of the ICD-10 codes in each sequence are masked out and the model needs to predict the original ICD-10 code of the masked token relying only on the surrounding context of codes.
* The code to train the model can be found in: [mlm_pre_trainer.py](mlm_pre_trainer.py)
* The training configurations can be found here: [config_files_emnlp/train/train.json](config_files_emnlp/train/train.json)
* The command to create the tokenizer and vocabulary is given in: [run_emnlp/tokenizer](run_emnlp/tokenizer)
* We used deepspeed while training our model, and the command is given in: [run_emnlp/model/train.sh](run_emnlp/model/train.sh)
* We used the development set to select the best performing model during the course of training.
</details>

---