import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error
)

from .m2m_metrics import M2MMetrics


class M2MMultiLabelMetrics(M2MMetrics):

    def __init__(self, alpha, do_lm, threshold):
        super().__init__(do_lm)
        self._alpha = alpha
        self._threshold = threshold

    def compute(self, model_output):
        if self._do_lm:
            return {**self.compute_lm_metrics(model_output), **self.compute_multi_label_metrics(model_output)}
        else:
            return {**self.compute_metrics(model_output), **self.compute_multi_label_metrics(model_output)}

    def compute_multi_label_metrics(self, model_output):
        # Get stuff
        model_output = model_output.predictions
        text_features = model_output[0]
        label_features = model_output[1]
        # Convert smoothed labels back to one hot encoded labels
        labels = (label_features >= (1 - self._alpha)).astype(int)
        # Apply sigmoid function to get prediction scores
        scores = expit(text_features)
        # Convert probabilities into predictions
        predictions = (scores >= self._threshold).astype(int)

        # Compute metrics
        precision = precision_score(y_true=labels, y_pred=predictions, average='micro')
        recall = recall_score(y_true=labels, y_pred=predictions, average='micro')
        f1 = f1_score(y_true=labels, y_pred=predictions, average='micro')
        roc_auc = roc_auc_score(y_true=labels, y_score=scores, average='micro')
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        cosine_similarity = (
                np.sum(scores * label_features, axis=1) /
                (np.linalg.norm(scores, axis=1) * np.linalg.norm(label_features, axis=1))
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc':roc_auc,
            'accuracy':accuracy,
            'mse': mean_squared_error(scores, label_features),
            'cosine_sim': np.mean(cosine_similarity),
        }
