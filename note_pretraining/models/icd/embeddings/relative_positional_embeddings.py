import torch


class RelativePositionalEmbeddings(object):
    """
    Class to handle the creation of positional embeddings for the model
    """

    @staticmethod
    def _angle_definition(pos: torch.FloatTensor, i: torch.LongTensor, d_model_size: int) -> torch.FloatTensor:
        """
        Compute the value for a given position id, and the value on which we apply the sin and cosine functions

        Args:
            pos (torch.FloatTensor): The tensor containing the position ids
            i (torch.LongTensor): A tensor that represents the embedding size with values (0, 1, 2 ..) that represent
            the position in the embedding vector
            d_model_size (int): The maximum length of the input position ids in the model

        Returns:
            (torch.FloatTensor): Values for each position in the embedding vector on which the sin and cosine operations
            are applied

        """
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
        angle_rates = angle_rates.to(device=pos.device)
        return pos * angle_rates

    @staticmethod
    def _relative_embeddings(
            position_ids: torch.FloatTensor,
            embedding_size: int,
            d_model_size: int,
            dtype: torch.dtype
    ):
        """
        Return the relative positional embedding vector for a given position id
        Args:
            position_ids (torch.FloatTensor): The position ids
            embedding_size (int): The size of the embeddings vector for each positional id
            d_model_size (int): The maximum length of the input position ids in the model
            dtype (int): The datatype of the tensors used in some calculations

        Returns:
            (torch.FloatTensor): Tensor containing the embedding vector for each positional id

        """
        # Create the sinusoidal pattern for the positional encoding
        angle_rads = RelativePositionalEmbeddings._angle_definition(
            position_ids.unsqueeze(2),
            torch.arange(embedding_size, dtype=dtype).unsqueeze(0),
            d_model_size,
        )
        # Apply the sin and cosine operations depending on the position of the embedding vector
        # Sin is applied on the even positions and cosine is applied on the odd positions
        angle_rads[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
        angle_rads[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])

        return angle_rads

    @staticmethod
    def positional_encoding(
            position_ids: torch.FloatTensor,
            embedding_size: int,
            d_model_size: int,
            dtype: torch.dtype,
    ) -> torch.FloatTensor:
        """
        Given a tensor of position ids return the positional embeddings for the corresponding position ids
        If the position embedding type is relative, the returned positional embeddings are based on the
        sinusoidal embeddings from the attention is all you need paper (). The position id values are used
        directly when generating the embeddings. Absolute positional embeddings are not yet implemented.

        Args:
            position_ids (torch.FloatTensor): The position ids
            embedding_size (int): The size of the embeddings vector for each positional id
            d_model_size (int): The maximum number of positional ids supported by the model (length of input)
            dtype (int): The datatype of the tensors used in some calculations

        Returns:
            (torch.FloatTensor): Tensor containing the embedding vector for each positional id

        Raises:
            (NotImplementedError): If position embedding type is absolute

        """
        return RelativePositionalEmbeddings._relative_embeddings(
            position_ids=position_ids,
            embedding_size=embedding_size,
            d_model_size=d_model_size,
            dtype=dtype
        )
