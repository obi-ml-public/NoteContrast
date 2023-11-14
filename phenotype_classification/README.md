# ICD Coding

**Pre-trained language models for automatic ICD-10 coding**

*Version 1.0.0 / 20 June 2022*

*We're working on expanding the documentation and readme. While we work on that, taking a look at the source code may help answer some questions.*

*Comments, feedback and improvements are welcome and encouraged!*

---

# Project Details 

*This section provides information about the project. This includes information about installation, project features, and a general project overview.*
*It is useful for getting a brief understanding of the project and submitting any questions to the project team*

<details>
<summary>Click to expand</summary>

* This repository was used to train models to perform the automatic ICD coding task.
* We test the models on the MIMIC-III-50, MIMIC-III-rare50 and MIMIC-III-full task.
* We treat the multi label task as a multi-class classification task using prompt based fine-tuning.
* Prompt-based fine-tuning is an alternative approach to multi-label classification where the multi-label classification task is reformulated as a cloze task.
</details>

---

# Dataset Creation
*This section provides information on how the dataset of medical text was created*

<details>
<summary>Click to expand</summary>

* We follow the steps presented in [CAML-MIMIC](https://github.com/jamesmullenbach/caml-mimic) & [KEPT](https://github.com/whaleloops/KEPT) to create the datasets for each task.
</details>

---

# Training
*This section provides information about the training data, the speed and size of training elements.*
*It is useful for people who want to learn more about the model inputs, objective, architecture, development, training/evaluation/prediction procedure and compute infrastructure.*

<details>
<summary>Click to expand</summary>

* We fine-tune 4 models on each task. * We used the development set to select the best performing model during the course of training. We fine-tune 5 replicates with 5 different random seeds:
  * NoteLM 4k - 4096 context length pre-trained with masked language modeling (MLM). 
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/train/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/train/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/train/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/train/full/)
  * NoteContrast 4k - 4096 context length pre-trained with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/train/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/train/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/train/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/train/full/)
  * NoteContrast 8k - 8192 context length pre-trained with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/train/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/train/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/train/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/train/full/)
  * NoteContrast 8k ICD - NoteContrast 8k fine-tuned on ICD-10 code descriptions with contrastive loss.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/train/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/train/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/train/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/train/full/)
    * MIMIC-III-full configurations: [config_files_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/train/full/](config_files_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/train/full/)
* The code for fine-tuning the models on the MIMIC-III-50 and MIMIC-III-rare50 is located here: [prompt_text_classifier.py](prompt_text_classifier.py)
* The code for fine-tuning the models on the MIMIC-III-full is located here: [mimic_full_classifier.py](mimic_full_classifier.py)
* We used deepspeed while training our model, and the commands for training all the models are structured as follows:
  * NoteLM 4k - 4096 context length pre-trained with masked language modeling (MLM). 
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/train.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/train.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/train.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/train.sh)
  * NoteContrast 4k - 4096 context length pre-trained with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/train.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/train.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/train.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/train.sh)
  * NoteContrast 8k - 8192 context length pre-trained with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/train.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/train.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/train.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/train.sh)
  * NoteContrast 8k ICD - NoteContrast 8k fine-tuned on ICD-10 code descriptions with contrastive loss.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/train.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/train.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/train.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/train.sh)
    * MIMIC-III-full configurations: [run_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/train.sh](run_emnlp/icd_classification_v1/mimic_full/mimic_full.sh)



</details>

---

# Evaluation
*This section describes the evaluation protocols and provides the results of the trained models.*

<details>
<summary>Click to expand</summary>

* We report micro and macro averaged F1 scores, micro and macro averaged AUC scores, precision at K(K={5,8,15}),and recall at K(K={8,15}). 
* All experiments were repeated 5 times with different random seeds (including model fine-tuning). 
* We used the development set to select the best performing model during the course of training.
* The best thresholds for classification and 417 computing precision, recall, and F1 were selected using the development set for each task.
* The evaluation configuration parameters can be found here:
  * NoteLM 4k - 4096 context length pre-tested with masked language modeling (MLM). 
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/test/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/test/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/test/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/test/full/)
  * NoteContrast 4k - 4096 context length pre-tested with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/test/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/test/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/test/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/test/full/)
  * NoteContrast 8k - 8192 context length pre-tested with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/test/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/test/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/test/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/test/full/)
  * NoteContrast 8k ICD - NoteContrast 8k fine-tuned on ICD-10 code descriptions with contrastive loss.
    * MIMIC-III-50 configurations: [config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/test/full/](config_files_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/test/full/)
    * MIMIC-III-rare50 configurations: [config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/test/full/](config_files_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/test/full/)
    * MIMIC-III-full configurations: [config_files_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/test/full/](config_files_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/test/full/)
* We use these configurations to run our evaluation on the three tasks and store the evaluation metrics using the commands present in:
  * NoteLM 4k - 4096 context length pre-trained with masked language modeling (MLM). 
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/test.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_lm_4096/test.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/test.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_lm_4096/test.sh)
  * NoteContrast 4k - 4096 context length pre-tested with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/test.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_4096/test.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/test.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_4096/test.sh)
  * NoteContrast 8k - 8192 context length pre-tested with contrastive loss and MLM.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/test.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_8192/test.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/test.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_8192/test.sh)
  * NoteContrast 8k ICD - NoteContrast 8k fine-tuned on ICD-10 code descriptions with contrastive loss.
    * MIMIC-III-50 configurations: [run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/test.sh](run_emnlp/icd_classification_v1/mimic_common_50/mimic_clip_lm_icd/test.sh)
    * MIMIC-III-rare50 configurations: [run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/test.sh](run_emnlp/icd_classification_v1/mimic_rare_50/mimic_clip_lm_icd/test.sh)
    * MIMIC-III-full configurations: [run_emnlp/icd_classification_v1/mimic_full/mimic_clip_lm_icd/test.sh](run_emnlp/icd_classification_v1/mimic_full/mimic_full.sh)

</details>

---