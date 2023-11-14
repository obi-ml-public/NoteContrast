import numpy as np
from typing import Optional

from datasets import DatasetDict, Dataset


class ClassificationDataset(object):
    """
    Class that outlines the functions that can be used to prepare the datasets
    for training and evaluation, and run the training and evaluation of the models
    """

    def __init__(
            self,
            text_datasets: DatasetDict,
    ):
        """
        Initialize the Huggingface DatasetDict object that contains the train, eval and test splits

        Args:
            text_datasets (DatasetDict): The dataset object that contains the data for each split (train, val etc)
        """

        self._text_datasets = text_datasets

    def get_train_dataset(self, max_train_samples: Optional[int]) -> Optional[Dataset]:
        """
        Return the train split of the dataset object.

        Args:
            max_train_samples (Optional[int]): For debugging purposes or quicker training, truncate the number of
            training examples to this value if set.

        Returns:
            train_dataset (Optional[Dataset]): The training dataset. Returns None if train split does not exist.
        """

        # Check if the DatasetDict object contains the train split
        if "train" not in self._text_datasets:
            return None
        train_dataset = self._text_datasets["train"]
        # Select only a subset of training examples if desired.
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        return train_dataset

    def get_eval_dataset(
            self,
            max_eval_samples: Optional[int],
            shuffle: bool = False,
            seed: int = 41
    ) -> Optional[Dataset]:
        """
        Return the validation split of the dataset object.

        Args:
            max_eval_samples (Optional[int]): For debugging purposes or quicker training, truncate the number of
            validation examples to this value if set.
            shuffle (bool, defaults to `True`): Shuffle the validation dataset.
            seed (int, defaults to `41`): Reproducible seed.

        Returns:
            eval_dataset (Optional[Dataset]): The validation dataset. Returns None if validation split does not exist.
        """

        # Check if the DatasetDict object contains the validation split
        if "validation" not in self._text_datasets:
            return None
        # Select only a subset of validation examples if desired.
        eval_dataset = self._text_datasets["validation"]
        if max_eval_samples is not None:
            if shuffle:
                np.random.seed(seed)
                max_eval_samples = min(len(eval_dataset), max_eval_samples)
                indexes = np.arange(len(eval_dataset))
                select_indexes = np.random.choice(indexes, max_eval_samples, replace=False)
                eval_dataset = eval_dataset.select(select_indexes)
            else:
                max_eval_samples = min(len(eval_dataset), max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
        return eval_dataset

    def get_test_dataset(self, max_test_samples: Optional[int]) -> Optional[Dataset]:
        """
        Return the test split of the dataset object.

        Args:
            max_test_samples (Optional[int]): For debugging purposes or quicker training, truncate the number of test
            examples to this value if set.

        Returns:
            test_dataset (Optional[Dataset]): The test dataset. Returns None if test split does not exist.
        """

        # Check if the DatasetDict object contains the test split
        if "test" not in self._text_datasets:
            return None
        test_dataset = self._text_datasets["test"]
        # Select only a subset of validation examples if desired.
        if max_test_samples is not None:
            max_test_samples = min(len(test_dataset), max_test_samples)
            test_dataset = test_dataset.select(range(max_test_samples))
        return test_dataset
