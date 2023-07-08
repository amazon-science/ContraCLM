# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import os


class Model(nn.Module):
    def __init__(self, encoder, pad_token_id=1):
        super(Model, self).__init__()
        self.encoder = encoder
        self.pad_token_id = pad_token_id

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            attention_mask = code_inputs.ne(self.pad_token_id)
            outputs = self.encoder(code_inputs, attention_mask=attention_mask)[0]
            outputs = (outputs * attention_mask[:, :, None]).sum(1) / attention_mask.sum(1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            attention_mask = nl_inputs.ne(self.pad_token_id)
            outputs = self.encoder(nl_inputs, attention_mask=attention_mask)[0]
            outputs = (outputs * attention_mask[:, :, None]).sum(1) / attention_mask.sum(1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

    def load_encoder_state_dict(self, encoder_state_dict, logger):
        if encoder_state_dict and os.path.exists(encoder_state_dict):
            state_dict = get_unwrapped_state_dict(encoder_state_dict, logger)
            self.encoder.load_state_dict(state_dict)


def get_unwrapped_state_dict(
        state_dict_path,
        logger,
        state_dict_attribute="auto",
        key_prefix="auto",
):
    unwrapped_state_dict = None
    if state_dict_path:
        logger.info(
            f"loading the model parameters from the ckpt file {state_dict_path!r}"
        )
        state_dict = torch.load(state_dict_path, map_location="cpu")
        if state_dict_attribute == "auto":
            if "state_dict" in state_dict:
                state_dict_attribute = "state_dict"
            elif "module" in state_dict:
                state_dict_attribute = "module"
            else:
                logger.warning(
                    "Unable to induce `state_dict_attribute`. Default to `None`."
                )
                state_dict_attribute = None
        if key_prefix == "auto":
            if "state_dict" in state_dict:
                key_prefix = "model.transformer."
            elif "module" in state_dict:
                key_prefix = "module.model.transformer."
            else:
                logger.warning("Unable to induce `key_prefix`. Default to `None`.")
                key_prefix = None
        if state_dict_attribute:
            logger.info(f"using state dict attribute {state_dict_attribute!r}")
            state_dict = state_dict[state_dict_attribute]
        if key_prefix:
            logger.info(f"using state dict key prefix {key_prefix!r}")
            unwrapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(key_prefix):
                    new_key = key[len(key_prefix):]
                    unwrapped_state_dict[new_key] = value
        else:
            unwrapped_state_dict = state_dict

    return unwrapped_state_dict
