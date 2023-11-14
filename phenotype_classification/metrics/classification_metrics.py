import re
import numpy as np

class ClassificationMetrics(object):

    def __init__(
            self,
            post_process,
            label_list,
            evaluator,
            optimal_f1_threshold=None
    ):
        self._post_process = post_process
        self._label_list = label_list
        self._evaluator = evaluator
        self._optimal_f1_threshold = optimal_f1_threshold

    def compute_binary_metrics(self, model_output):

        if self._optimal_f1_threshold:
            prediction_scores, predictions, labels = self._post_process.decode(
                model_predictions=model_output.predictions, model_labels=model_output.label_ids
            )
            optimal_f1_threshold = self._evaluator.get_optimal_threshold(
                prediction_scores,
                labels
            )
            predictions = (np.array(prediction_scores) >= optimal_f1_threshold).astype(int)

        else:
            prediction_scores, predictions, labels = self._post_process.decode(
                model_predictions=model_output.predictions, model_labels=model_output.label_ids
            )
            optimal_f1_threshold = -100

        metrics_report = self.get_metrics_report(
            prediction_scores=prediction_scores,
            predictions=predictions,
            labels=labels
        )

        results = self.get_metrics_dictionary(metrics_report=metrics_report)

        results = self.format_results(
            results={
                **results,
                **{'optimal_threshold':optimal_f1_threshold}
            }
        )

        results = {
            re.sub(r'(_)?pos(_)?', '', key): value
            for key, value in results.items()
        }

        return results

    def compute_multi_class_metrics(self, model_output):

        prediction_scores, predictions, labels = self._post_process.decode(
            model_predictions=model_output.predictions, model_labels=model_output.label_ids
        )

        metrics_report = self.get_metrics_report(
            predictions=predictions,
            prediction_scores=prediction_scores,
            labels=labels
        )

        results = self.get_metrics_dictionary(metrics_report=metrics_report)

        return self.format_results(results=results)

    def compute_multi_label_metrics(self, model_output):

        if self._optimal_f1_threshold:
            prediction_scores, predictions, labels = self._post_process.decode(
                model_predictions=model_output.predictions, model_labels=model_output.label_ids
            )
            optimal_f1_threshold = self._evaluator.get_optimal_threshold(
                prediction_scores, labels
            )
            predictions = (np.array(prediction_scores) > optimal_f1_threshold).astype(int)
        else:
            prediction_scores, predictions, labels = self._post_process.decode(
                model_predictions=model_output.predictions, model_labels=model_output.label_ids
            )
            optimal_f1_threshold = -100

        metrics_report = self.get_metrics_report(
            prediction_scores=prediction_scores,
            predictions=predictions,
            labels=labels
        )

        results = self.get_metrics_dictionary(metrics_report=metrics_report)

        return self.format_results(
            results={**results, **{'optimal_threshold':optimal_f1_threshold}}
        )


    def get_metrics_report(self, prediction_scores, predictions, labels):
        return self._evaluator.classification_report(
            labels=labels,
            predictions=predictions,
            prediction_scores=prediction_scores,
            label_list=self._label_list
        )

    @staticmethod
    def get_auc_score(auc_score):
        if auc_score is not None:
            return {'auc': auc_score}
        else:
            return {}

    @staticmethod
    def get_auprc_score(auprc_score):
        if auprc_score is not None:
            return {'auprc': auprc_score}
        else:
            return {}

    @staticmethod
    def get_mcc_score(mcc_score):
        if mcc_score is not None:
            return {'mcc': mcc_score}
        else:
            return {}

    @staticmethod
    def get_micro_avg_score(micro_score):
        if micro_score is not None:
            # Extract micro averaged token level score
            return {
                'micro_avg': {
                    "precision": micro_score["precision"],
                    "recall": micro_score["recall"],
                    "f1": micro_score["f1-score"]}
            }
        else:
            return {}

    @staticmethod
    def get_macro_avg_score(macro_score):
        if macro_score is not None:
            # Extract macro averaged token level score
            return {
                'macro_avg': {
                    "precision": macro_score["precision"],
                    "recall": macro_score["recall"],
                    "f1": macro_score["f1-score"]}
            }
        else:
            return {}

    @staticmethod
    def get_weighted_avg_score(weighted_score):
        if weighted_score is not None:
            # Extract weighted averaged token level score
            return {
                'weighted_avg': {
                    "precision": weighted_score["precision"],
                    "recall": weighted_score["recall"],
                    "f1": weighted_score["f1-score"]}
            }
        else:
            return {}

    @staticmethod
    def get_precision_at_k_score(precision_at_k_score):
        if precision_at_k_score is not None:
            return {'precision_at': precision_at_k_score}
        else:
            return {}

    @staticmethod
    def get_recall_at_k_score(recall_at_k_score):
        if recall_at_k_score is not None:
            return {'recall_at': recall_at_k_score}
        else:
            return {}


    def get_metrics_dictionary(self, metrics_report):

        micro_overall = self.get_micro_avg_score(metrics_report.pop("micro avg", None))
        macro_overall = self.get_macro_avg_score(metrics_report.pop("macro avg", None))
        weighted_overall = self.get_weighted_avg_score(metrics_report.pop("weighted avg", None))
        auc_overall = self.get_auc_score(metrics_report.pop("auc", None))
        auprc_overall = self.get_auprc_score(metrics_report.pop("auprc", None))
        mcc_overall = self.get_mcc_score(metrics_report.pop("matthews_correlation_coefficient", None))
        precision_at_k_overall = self.get_precision_at_k_score(metrics_report.pop("precision_at", None))
        recall_at_k_overall = self.get_recall_at_k_score(metrics_report.pop("recall_at", None))

        # Extract token level scores for each NER type
        scores = {
            type_name.lower(): {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in metrics_report.items()
        }

        return {
            **scores,
            **micro_overall,
            **macro_overall,
            **weighted_overall,
            **auc_overall,
            **auprc_overall,
            **mcc_overall,
            **precision_at_k_overall,
            **recall_at_k_overall
        }

    @staticmethod
    def format_results(results):
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
