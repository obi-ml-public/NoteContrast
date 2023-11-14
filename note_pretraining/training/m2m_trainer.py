from typing import Dict, Union, Any, Optional, List, Tuple

import torch
from torch import nn
from transformers import Trainer
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach


class M2MTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(
                tuple(
                    inputs.get(name).get(
                        'labels'
                    ) for name in self.label_names if inputs.get(name).get('labels', None) is not None
                )
            )
            if len(labels) == 1:
                labels = labels[0]

        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError('Import errors are preventing this section of code from working')
                # raw_outputs = smp_forward_only(model, inputs)
                # if has_labels:
                #     if isinstance(raw_outputs, dict):
                #         loss_mb = raw_outputs["loss"]
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                #     else:
                #         loss_mb = raw_outputs[0]
                #         logits_mb = raw_outputs[1:]
                #
                #     loss = loss_mb.reduce_mean().detach().cpu()
                #     logits = smp_nested_concat(logits_mb)
                # else:
                #     loss = None
                #     if isinstance(raw_outputs, dict):
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                #     else:
                #         logits_mb = raw_outputs
                #     logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return loss, logits, labels
