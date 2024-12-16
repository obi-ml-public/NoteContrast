# Overview

* The _icd_embeddings_ directory contains code to train contextual embedding models for ICD-10 diagnostic codes based on a large real-world data set.
* The _note_pretraining_ directory was used to pre-train medical text models using masked language modeling, contrastive learning or both.
* The _phenotype_classification_ folder contains code to perform the automatic ICD coding task, and evaluations on MIMIC-III.

# Citing

If you found this repository useful, please consider citing:

```
@InProceedings{pmlr-v225-kailas23a,
  title = 	 {NoteContrast: Contrastive Language-Diagnostic Pretraining for Medical Text},
  author =       {Kailas, Prajwal and Homilius, Max and Deo, Rahul C. and MacRae, Calum A.},
  booktitle = 	 {Proceedings of the 3rd Machine Learning for Health Symposium},
  pages = 	 {201--216},
  year = 	 {2023},
  editor = 	 {Hegselmann, Stefan and Parziale, Antonio and Shanmugam, Divya and Tang, Shengpu and Asiedu, Mercy Nyamewaa and Chang, Serina and Hartvigsen, Tom and Singh, Harvineet},
  volume = 	 {225},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v225/kailas23a/kailas23a.pdf},
  url = 	 {https://proceedings.mlr.press/v225/kailas23a.html},
}
```


