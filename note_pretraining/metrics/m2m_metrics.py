import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error
metric = evaluate.load("accuracy")

class M2MMetrics(object):

    def __init__(self, do_lm):
        self._do_lm = do_lm

    def compute(self, model_output):
        if self._do_lm:
            return self.compute_lm_metrics(model_output)
        else:
            return self.compute_metrics(model_output)

    def compute_lm_metrics(self, model_output):
        # lm_labels = model_output.label_ids
        model_output = model_output.predictions
        # Get stuff
        text_features = model_output[0]
        label_features = model_output[1]
        text_lm_loss = np.mean(model_output[3])
        m2m_loss = np.mean(model_output[4])
        # lm_accuracy = self.compute_mlm_accuracy(predictions=model_output[5], labels=lm_labels)
        lm_accuracy =  0
        cosine_similarity = (
                np.sum(text_features * label_features, axis=1) /
                (np.linalg.norm(text_features, axis=1) * np.linalg.norm(label_features, axis=1))
        )
        return {
            'mse': mean_squared_error(text_features, label_features),
            'cosine_sim': np.mean(cosine_similarity),
            'm2m_loss': np.mean(m2m_loss),
            'text_lm_loss': text_lm_loss,
            'text_lm_perplexity': np.exp(text_lm_loss),
            'lm_accuracy': lm_accuracy,
            'logit_scale': model_output[2][0]
        }

    @staticmethod
    def compute_metrics(model_output):
        model_output = model_output.predictions
        text_features = model_output[0]
        label_features = model_output[1]
        cosine_similarity = (
                np.sum(text_features * label_features, axis=1) /
                (np.linalg.norm(text_features, axis=1) * np.linalg.norm(label_features, axis=1))
        )
        return {
            'mse': mean_squared_error(text_features, label_features),
            'cosine_sim': np.mean(cosine_similarity),
            'logit_scale': model_output[2][0]
        }

    def compute_mlm_accuracy(self, predictions, labels):
        predictions = self.preprocess_logits_for_metrics(predictions)
        # predictions have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        return metric.compute(predictions=predictions, references=labels)['accuracy']

    @staticmethod
    def preprocess_logits_for_metrics(logits):
        return np.argmax(logits, axis=2)