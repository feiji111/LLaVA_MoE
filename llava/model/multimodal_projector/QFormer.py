from typing import Optional

from transformers import Blip2QFormerConfig, Blip2QFormerModel
from transformers.modeling_utils import PreTrainedModel
from .configuration import QFormerConfig

import torch
from torch import nn

class QFormer(Blip2QFormerModel):
    def __init__(self, config: QFormerConfig):
        super().__init__(config.qformer_config)
        self.num_query_tokens = config.num_query_tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))

    def forward(
        self, 
        image_embeds: Optional[torch.FloatTensor] = None,
        image_embeds_multi: Optional[torch.FloatTensor] = None, #only used for tokenpacker
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = super().forward(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = query_outputs.last_hidden_states

        return last_hidden_states
