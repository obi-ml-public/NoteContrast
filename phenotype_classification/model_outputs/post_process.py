import numpy as np
from typing import Sequence, Tuple, Any


class Process(object):
    """
    Process the output of the model forward pass. The forward pass will return the predictions
    (e.g. the logits), labels if present. We process the output and return the processed
    values based on the application.
    """

    def __init__(
            self,
            label_list: Sequence[str],
            threshold=None
    ):
        """
        Initialize the variables

        Args:
            label_list (Sequence[str]): A label list where the position corresponds to a particular label. For example
            position 0 will correspond to B-DATE etc.
        """

        self._label_list = label_list
        self._threshold = threshold

    def get_threshold(self):
        return self._threshold

    def get_predictions(
            self,
            model_predictions: np.array
    ) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.
        """

        raise NotImplementedError('Use subclass method')

    def get_labels(self, model_labels: np.array) -> np.array:
        raise NotImplementedError('Use subclass method')

    def decode(
            self,
            model_predictions: np.array,
            model_labels: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc.) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions.
        Also remove the predictions and labels on the subword and context tokens.

        Args:
            model_predictions (np.array): The logits (scores for each tag)
            returned by the model.
            model_labels (np.array): Gold standard labels.

        Returns:
            true_predictions (np.array): The predicted NER tags.
            true_labels (np.array): The gold standard NER tags.
        """

        raise NotImplementedError('Use subclass method')



class BinaryProcess(Process):
    """
    Process the output of the model forward pass. Given the model logits (numerical prediction scores)
    we return the prediction of the model as the entity with the highest numerical score.
    """

    def __init__(self, label_list: Sequence[str], threshold=None):
        """
        Initialize the variables

        Args:
            label_list (Sequence[str]): A label list where the position corresponds to a particular label. For example
            position 0 will correspond to B-DATE etc.
        """

        super().__init__(label_list=label_list, threshold=threshold)

    @staticmethod
    def get_prediction_scores(
            model_predictions: np.array
    ) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """

        return model_predictions[:, 1] - model_predictions[:, 0]

    def get_predictions(self, model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """
        if self._threshold is None:
            return np.argmax(model_predictions, axis=1)
        else:
            prediction_scores = self.get_prediction_scores(model_predictions)
            return (prediction_scores >= self._threshold).astype(int)

    def get_labels(self, model_labels):
        return model_labels

    def decode(
            self,
            model_predictions: np.array,
            model_labels: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc.) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions.
        Also remove the predictions and labels on the subword and context tokens.

        Args:
            model_predictions (np.array): The logits (scores for each tag)
            returned by the model.
            model_labels (np.array): Gold standard labels.

        Returns:
            true_predictions (np.array): The predicted NER tags.
            true_labels (np.array): The gold standard NER tags.
        """

        prediction_scores = self.get_prediction_scores(model_predictions=model_predictions)
        predictions = self.get_predictions(model_predictions=model_predictions)
        labels = self.get_labels(model_labels=model_labels)

        return prediction_scores, predictions, labels

class MultiClassProcess(Process):
    """
    Process the output of the model forward pass. Given the model logits (numerical prediction scores)
    we return the prediction of the model as the entity with the highest numerical score.
    """

    def __init__(self, label_list: Sequence[str], threshold=None):
        """
        Initialize the variables

        Args:
            label_list (Sequence[str]): A label list where the position corresponds to a particular label. For example
            position 0 will correspond to B-DATE etc.
        """

        super().__init__(label_list=label_list, threshold=threshold)

    @staticmethod
    def get_prediction_scores(
            model_predictions: np.array
    ) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """

        return model_predictions

    def get_predictions(self, model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """
        return np.argmax(model_predictions, axis=1)

    def get_labels(self, model_labels):
        return model_labels

    def decode(
            self,
            model_predictions: np.array,
            model_labels: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc.) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions.
        Also remove the predictions and labels on the subword and context tokens.

        Args:
            model_predictions (np.array): The logits (scores for each tag)
            returned by the model.
            model_labels (np.array): Gold standard labels.

        Returns:
            true_predictions (np.array): The predicted NER tags.
            true_labels (np.array): The gold standard NER tags.
        """

        prediction_scores = self.get_prediction_scores(model_predictions=model_predictions)
        predictions = self.get_predictions(model_predictions=model_predictions)
        labels = self.get_labels(model_labels=model_labels)

        return prediction_scores, predictions, labels

class PromptArgmaxProcess(Process):
    """
    Process the output of the model forward pass. Given the model logits (numerical prediction scores)
    we return the prediction of the model as the entity with the highest numerical score.
    """

    def __init__(
            self,
            label_list: Sequence[str],
            ignore_label=-100,
            threshold=None
    ):
        """
        Initialize the variables

        Args:
            label_list (Sequence[str]): A label list where the position corresponds to a particular label. For example
            position 0 will correspond to B-DATE etc.
        """

        super().__init__(
            label_list=label_list,
            threshold=threshold
        )
        self._ignore_label = ignore_label

    def filter_by_dataset_labels(
            self,
            dataset_lists: np.array,
            dataset_labels: np.array
    ) -> Sequence[Sequence[Any]]:
        """
        Filter any dataset list based on the dataset labels. Remove elements in the
        dataset lists where the corresponding dataset label is the token_ignore_label
        which by default has the value 'NA'.

        Args:
            dataset_lists (np.array): A sequence of data to be filtered
            dataset_labels (np.array): The corresponding list that is used to filter that dataset list.
            This list should contain the token ignore label.

        Returns:
            (Sequence[Sequence[Any]]): The filtered dataset list.
        """

        return [
            [dataset_item for (dataset_item, label) in zip(dataset_list, labels) if label != self._ignore_label]
            for dataset_list, labels in zip(dataset_lists, dataset_labels)
        ]

    @staticmethod
    def get_prediction_scores(model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """

        return model_predictions[:, :, 1] -  model_predictions[:, :, 0]

    def get_predictions(self, model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """

        if self._threshold is None:
            return np.argmax(model_predictions, axis=2)
        else:
            prediction_scores = self.get_prediction_scores(model_predictions)
            return (prediction_scores >= self._threshold).astype(int)

    def get_labels(self, model_labels):
        return model_labels

    def decode(
            self,
            model_predictions: np.array,
            model_labels: np.array
    ) -> Tuple[Sequence[Sequence[int]], Sequence[Sequence[float]], Sequence[Sequence[int]]]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc.) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions.
        Also remove the predictions and labels on the subword and context tokens.

        Args:
            model_predictions (np.array): The logits (scores for each tag)
            returned by the model.
            model_labels (np.array): Gold standard labels.

        Returns:
            true_predictions (np.array): The predicted NER tags.
            true_labels (np.array): The gold standard NER tags.
        """

        prediction_scores = self.get_prediction_scores(model_predictions=model_predictions)
        predictions = self.get_predictions(model_predictions=model_predictions)
        labels = self.get_labels(model_labels=model_labels)

        prediction_scores = self.filter_by_dataset_labels(
            dataset_lists=prediction_scores, dataset_labels=labels
        )
        predictions = self.filter_by_dataset_labels(
            dataset_lists=predictions, dataset_labels=labels
        )
        labels = self.filter_by_dataset_labels(
            dataset_lists=labels, dataset_labels=labels
        )

        return prediction_scores, predictions, labels


class MultiLabelProcess(Process):
    """
    Process the output of the model forward pass. Given the model logits (numerical prediction scores)
    we return the prediction of the model as the entity with the highest numerical score.
    """

    def __init__(
            self,
            label_list: Sequence[str],
            threshold=None
    ):
        """
        Initialize the variables

        Args:
            label_list (Sequence[str]): A label list where the position corresponds to a particular label. For example
            position 0 will correspond to B-DATE etc.
        """

        super().__init__(label_list=label_list, threshold=threshold)


    def get_predictions(self, model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """
        if self._threshold is None:
            return np.round((np.sign(model_predictions) + 1) / 2)
        else:
            raise NotImplementedError()

    @staticmethod
    def get_prediction_scores(model_predictions: np.array) -> np.array:
        """
        Function to process the model predictions (the logits/scores specifically) and return the model
        predictions - i.e. return the actual ner tag indexes the model predicted. The index of the entity
        with the highest numerical score is returned as the prediction.

        Args:
            model_predictions (np.array): The numerical predictions made by the model.

        Returns:
            (np.array): The predicted NER indexes.
        """

        return model_predictions

    def get_labels(self, model_labels):
        return model_labels

    def decode(
            self,
            model_predictions: np.array,
            model_labels: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Decode the predictions and labels so that the evaluation function and prediction
        functions can use them accordingly. The predictions and labels are numbers (ids)
        of the labels, these will be converted back to the NER tags (B-AGE, I-DATE etc.) using
        the label_list. In this function we just take the argmax of the logits (scores) of the predictions.
        Also remove the predictions and labels on the subword and context tokens.

        Args:
            model_predictions (np.array): The logits (scores for each tag)
            returned by the model.
            model_labels (np.array): Gold standard labels.

        Returns:
            true_predictions (np.array): The predicted NER tags.
            true_labels (np.array): The gold standard NER tags.
        """
        prediction_scores = self.get_prediction_scores(model_predictions=model_predictions)
        predictions = self.get_predictions(model_predictions=model_predictions)
        labels = self.get_labels(model_labels=model_labels)

        return prediction_scores, predictions, labels
