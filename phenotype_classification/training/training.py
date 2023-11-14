import logging
from datasets import DatasetDict


class Training(object):
    """
    Class that outlines the functions that can be used to prepare the datasets
    for training and evaluation, and run the training and evaluation of the models
    """

    def __init__(
            self,
            classification_datasets: DatasetDict,
    ):
        """
        Initialize the huggingface to trainer to None
        Initialize the dataset object that contains the train and eval splits

        Args:
            classification_datasets (DatasetDict): The dataset object that contains the data for each split (train, val etc)
        """
        self._trainer = None
        self._sentence_datasets = classification_datasets

    def set_trainer(self, trainer):
        if self._trainer is None:
            logging.info("Initializing trainer")
        else:
            logging.warning("Trainer already initialized, re-initializing with new trainer")
        self._trainer = trainer

    def run_train(self, train_dataset, resume_from_checkpoint, last_checkpoint):
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = self._trainer.train(resume_from_checkpoint=checkpoint)
        self._trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)
        self._trainer.save_state()
        return metrics

    def run_eval(self, eval_dataset):
        logging.info("*** Evaluate ***")
        metrics = self._trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        self._trainer.log_metrics("eval", metrics)
        self._trainer.save_metrics("eval", metrics)
        return metrics

    def run_predict(self, test_dataset):
        return self._trainer.predict(test_dataset)