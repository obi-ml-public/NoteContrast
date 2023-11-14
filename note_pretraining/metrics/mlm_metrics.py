import evaluate
metric = evaluate.load("accuracy")

class MLMMetrics(object):

    @staticmethod
    def compute_metrics(model_output):
        predictions, labels = model_output
        # predictions have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        return metric.compute(predictions=predictions, references=labels)