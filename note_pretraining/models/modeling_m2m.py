import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .modeling_output import M2MModelOutput, M2MLMModelOutput


class M2MForPretrain(nn.Module):

    def __init__(self, m2m, m2m_loss):
        super().__init__()
        self._m2m = m2m
        self._m2m_loss = m2m_loss

    def forward(self, m2m_texts, m2m_labels, output_features=None, return_dict=True):

        if 'labels' in m2m_texts:
            m2m_texts.pop('labels')
        if 'labels' in m2m_labels:
            m2m_labels.pop('labels')

        # Get the model forward pass output
        text_features, label_features, _, _ = self._m2m(
            m2m_texts=m2m_texts,
            m2m_labels=m2m_labels,
            text_output_last_hidden_state=False,
            label_output_last_hidden_state=False
        )

        # Compute model to model loss
        m2m_loss = self._m2m_loss(text_features, label_features)

        # Gather the data we want to return
        return_features = (
            text_features, label_features, self._m2m_loss.get_logit_scale()
        ) if output_features else (None, None, None)
        if not return_dict:
            return ((m2m_loss, ) + return_features) if m2m_loss is not None else return_features

        # Return the model forward pass output
        return M2MModelOutput(
            loss=m2m_loss,
            text_features=text_features,
            label_features=label_features,
            logit_scale=self._m2m_loss.get_logit_scale(),
        )

    def get_m2m_text(self):
        return self._m2m

    @torch.jit.ignore
    def gradient_checkpointing_enable(self):
        self._m2m.gradient_checkpointing_enable()

    @torch.jit.ignore
    def gradient_checkpointing_disable(self):
        self._m2m.gradient_checkpointing_disable()


class M2MLMForPretrain(M2MForPretrain):

    def __init__(
            self,
            m2m,
            m2m_loss,
            text_lm_objective,
            text_lm_head,
            text_lm_vocab_size,
            loss_weighting,
    ):
        super().__init__(m2m, m2m_loss)
        self._text_lm_objective = text_lm_objective
        self._text_lm_head = text_lm_head
        self._text_lm_vocab_size = text_lm_vocab_size
        self._loss_weighting = loss_weighting

    def forward(self, m2m_texts, m2m_labels, output_features=None, return_dict=True):

        # Gather the language modeling task labels
        text_labels = None
        if 'labels' in m2m_texts:
            text_labels = m2m_texts.pop('labels')
        if 'labels' in m2m_labels:
            m2m_labels.pop('labels')

        # Get the model forward pass output
        text_features, label_features, text_sequence_output, label_sequence_output = self._m2m(
            m2m_texts=m2m_texts,
            m2m_labels=m2m_labels,
            text_output_last_hidden_state=False if text_labels is None else True,
            label_output_last_hidden_state=False
        )

        # Get the language modeling head output
        text_prediction_scores = self._text_lm_head(
            text_sequence_output
        ) if text_sequence_output is not None else None

        # Compute the language modeling loss
        text_lm_loss = None

        if text_labels is not None:
            if self._text_lm_objective == 'clm':
                text_lm_loss = self.__get_clm_loss(
                    prediction_scores=text_prediction_scores,
                    labels=text_labels,
                    vocab_size=self._text_lm_vocab_size
                )
            elif self._text_lm_objective == 'mlm':
                text_lm_loss = self.__get_mlm_loss(
                    prediction_scores=text_prediction_scores,
                    labels=text_labels,
                    vocab_size=self._text_lm_vocab_size
                )
            else:
                raise ValueError('Invalid LM objective specified')

        # Compute the model to model loss
        m2m_loss = self._m2m_loss(text_features, label_features)

        loss = self._loss_weighting(m2m_loss, text_lm_loss)

        # Gather the data we want to return
        return_features = (
            text_features, label_features, self._m2m_loss.get_logit_scale()
        ) if output_features else (None, None, None, None, None)

        if not return_dict:
            return ((loss,) + return_features + (text_lm_loss, m2m_loss)) if m2m_loss is not None else return_features

        # Return the model forward pass output
        return M2MLMModelOutput(
            loss=loss,
            text_features=text_features,
            label_features=label_features,
            logit_scale=self._m2m_loss.get_logit_scale(),
            text_lm_loss=text_lm_loss,
            m2m_loss=m2m_loss,
            text_logits=None,
            label_logits=None
        )

    def set_loss_weighting(self, loss_weighting):
        self._loss_weighting = loss_weighting

    def get_text_lm_head(self):
        return self._text_lm_head

    def get_label_lm_head(self):
        return self._label_lm_head

    @staticmethod
    def __get_clm_loss(prediction_scores, labels, vocab_size):
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        return loss_fct(
            shifted_prediction_scores.view(-1, vocab_size), labels.view(-1)
        )

    @staticmethod
    def __get_mlm_loss(prediction_scores, labels, vocab_size):
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        return loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))
