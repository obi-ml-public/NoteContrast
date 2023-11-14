from .prompt_label_transform import PromptLabelTransform

class DynamicPromptLabelTransform(PromptLabelTransform):
    """
    Class to handle dataset loading (from files and loading to dataset object) and dataset tokenization
    """

    def __init__(self, mask_token_id, label_to_id=None):
        super().__init__(label_to_id, mask_token_id)
        self._mask_token_id = mask_token_id

    def get_label_to_id(self, prompt_labels=None):
        label_to_id = {label: index_id for index_id, label in enumerate(prompt_labels)}
        return label_to_id

    # Local function to tokenize the inputs
    def tokenize_function(self, tokens_list, input_ids_list, prompt_labels_list=None):
        if isinstance(tokens_list[0], list):
            return [
                self.get_label_vector(
                    labels=labels,
                    input_ids=input_ids,
                    prompt_labels=prompt_labels
                ).tolist()
                for labels, input_ids, prompt_labels in zip(tokens_list, input_ids_list, prompt_labels_list)
            ]
        else:
            return self.get_label_vector(
                labels=tokens_list,
                input_ids=input_ids_list,
                prompt_labels=prompt_labels_list
            ).tolist()