import torch

def get_model_embedding(features, attention_mask, token_from_end):
    if token_from_end == -1:
        return features[:, 0, :]
    else:
        return features[
            torch.arange(features.shape[0]),
            attention_mask.sum(dim=1) - (token_from_end + 1)
        ]
