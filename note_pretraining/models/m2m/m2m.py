import torch
from torch import nn
from .heads import ClassificationHead, ProjectionHead
from .heads.utils import get_model_embedding

class M2M(nn.Module):
    def __init__(
            self,
            text_model,
            label_model,
            text_projection,
            label_projection,
            text_token_from_end,
            label_token_from_end,
            text_use_projection_head,
            label_use_projection_head
    ):
        super().__init__()
        self._text_model = text_model
        self._label_model = label_model
        self._label_token_from_end = label_token_from_end
        self._text_token_from_end = text_token_from_end
        self._text_use_projection_head = text_use_projection_head
        self._label_use_projection_head = label_use_projection_head

        self._text_head = self.get_head(
            config=self._text_model.config,
            projection=text_projection,
            use_projection_head=self._text_use_projection_head,
            token_from_end=self._text_token_from_end
        )
        self._label_head = self.get_head(
            config=self._label_model.config if hasattr(self._label_model, 'config') else None,
            projection=label_projection,
            use_projection_head=self._label_use_projection_head,
            token_from_end=self._label_token_from_end
        )

    @staticmethod
    def get_head(config, projection, use_projection_head, token_from_end):
        if projection:
            if use_projection_head:
                projection_head = ProjectionHead(
                    config=config,
                    projection=projection,
                    token_from_end=token_from_end
                )
                projection_head = projection_head.apply(projection_head.init_weights)
                return projection_head
            else:
                classification_head = ClassificationHead(
                    config=config,
                    projection=projection,
                    token_from_end=token_from_end
                )
                classification_head = classification_head.apply(classification_head.init_weights)
                return classification_head
        else:
            return None

    @staticmethod
    def encode_input(
            attention_mask,
            last_hidden_state,
            head,
            token_from_end,
            output_last_hidden_state
    ):

        if head is not None:
            cls_output = head(last_hidden_state)
        else:
            cls_output = get_model_embedding(
                features=last_hidden_state,
                attention_mask=attention_mask,
                token_from_end=token_from_end
            )

        if output_last_hidden_state:
            return cls_output, last_hidden_state
        else:
            return cls_output, None

    def encode_m2m_texts(self, m2m_texts, output_last_hidden_state):
        last_hidden_state = self._text_model(**m2m_texts).last_hidden_state
        return self.encode_input(
            attention_mask=m2m_texts['attention_mask'],
            last_hidden_state=last_hidden_state,
            head=self._text_head,
            token_from_end=self._text_token_from_end,
            output_last_hidden_state=output_last_hidden_state,
        )

    def encode_m2m_labels(self, m2m_labels, output_last_hidden_state):
        # If input is <s> a b c </s>
        # we want to select the output embedding from position c
        # Hence we take sum (to get rid of padding tokens - where attention mask is 0)
        # then we select the last element - ((label_token_from_end + 1))
        # This is done for every input in the batch
        last_hidden_state = self._label_model(**m2m_labels).last_hidden_state
        return self.encode_input(
            attention_mask=m2m_labels.get('attention_mask', None),
            last_hidden_state=last_hidden_state,
            head=self._label_head,
            token_from_end=self._label_token_from_end,
            output_last_hidden_state=output_last_hidden_state,
        )

    def forward(self, m2m_texts, m2m_labels, text_output_last_hidden_state=False, label_output_last_hidden_state=False):
        text_features, text_last_hidden_state = self.encode_m2m_texts(
            m2m_texts=m2m_texts, output_last_hidden_state=text_output_last_hidden_state
        )
        label_features, label_last_hidden_state = self.encode_m2m_labels(
            m2m_labels=m2m_labels, output_last_hidden_state=label_output_last_hidden_state
        )

        return (
            text_features,
            label_features,
            text_last_hidden_state,
            label_last_hidden_state,
        )

    def get_text_model(self):
        return self._text_model

    def get_label_model(self):
        return self._label_model

    def get_text_head(self):
        return self._text_head

    def get_label_head(self):
        return self._label_head

    @torch.jit.ignore
    def gradient_checkpointing_enable(self):
        self._text_model.gradient_checkpointing_enable()
        self._label_model.gradient_checkpointing_enable()

    @torch.jit.ignore
    def gradient_checkpointing_disable(self):
        self._text_model.gradient_checkpointing_disable()
        self._label_model.gradient_checkpointing_disable()
