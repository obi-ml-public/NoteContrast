import torch
from transformers import RobertaModel, RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from .embeddings import RelativePositionalEmbeddings


class ICDRobertaEmbeddings(RobertaEmbeddings):
    """
    SubClass the handles the creation of input embeddings for the model
    Input embeddings consist of word and positional embeddings. The parent
    class does not implement sinusoidal positional embeddings, this subclass lets you
    implements sinusoidal positional embeddings using position ids
    """

    def __init__(self, config):
        """
        Initialize the model config and also set position_embeddings to None.
        Position embeddings are set to None, since they are generated on the fly
        based on the position ids.

        Args:
            config:
        """
        super().__init__(config)
        # Set the position_embeddings of the parent class to None
        # We set the position embeddings based on the positions ids passed
        self.position_embeddings = None
        self.config = config

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)

        if position_ids is not None:
            # Get the position embedding vector for each position id
            position_embeddings = RelativePositionalEmbeddings.positional_encoding(
                position_ids=position_ids,
                embedding_size=self.config.hidden_size,
                d_model_size=self.config.max_position_embeddings,
                dtype=torch.long,
            )
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ICDRobertaModel(RobertaModel):
    """
    Subclass of RobertaModel that creates the input embeddings differently.
    Specifically uses ICDRobertaEmbeddings where sinusoidal position embeddings can be used.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # Use the embeddings based on ICDRobertaEmbeddings
        self.embeddings = ICDRobertaEmbeddings(config)
        # Initialize weights and apply final processing
        self.post_init()


class ICDRobertaForMaskedLM(RobertaForMaskedLM):
    """
    Subclass of RobertaForMaskedLM that creates the input embeddings differently.
    Specifically uses ICDRobertaEmbeddings where sinusoidal position embeddings can be used.
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = ICDRobertaModel(config, add_pooling_layer=False)
        # Initialize weights and apply final processing
        self.post_init()
