from typing import List, Dict


class LabelInfo(object):
    """

    """

    def __init__(self, label_list):
        """
        """

        self._label_list = label_list

    def get_label_list(self) -> List[str]:
        """
        """
        return self._label_list

    def get_label_to_id(self) -> Dict[str, int]:
        """
        Return a label to id mapping.

        Returns:
            label_to_id (Dict[str, int]): label to id mapping
        """

        labels = self.get_label_list()
        label_to_id = {label: index_id for index_id, label in enumerate(labels)}
        return label_to_id

    def get_id_to_label(self) -> Dict[int, str]:
        """
        Return an id to label mapping.

        Returns:
            id_to_label (Dict[int, str]): id to label mapping.
        """

        labels = self.get_label_list()
        id_to_label = {index_id: label for index_id, label in enumerate(labels)}
        return id_to_label
