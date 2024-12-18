import torch
from torch import nn
import torch.nn.functional as F
from .Qformer.Qformer import Qformer
from .TokenPacker import TokenPacker
from .MplugOwlVisualAbstractor import MplugOwlVisualAbstractorModel
from .configuration import MplugOwlVisualAbstractorConfig

from transformers import Blip2QFormerModel
from transformers.configuration_utils import PretrainedConfig
from .QFormer import QFormer
from .Mlp_GELU import Mlp_GELU

class MoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = 1  # top-1 expert per token (for pruning)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Define expert layers
        self.experts = nn.ModuleList([
            TokenPacker(scale_factor=4),  # Example expert, you can have different types
            Mlp_GELU("mlp2x_gelu")        # Another example expert
        ])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Process hidden states with expert routing """

        # hidden_states is a tuple (image_features, image_features_multi)
        hidden_states_multi = hidden_states[1]  # Used for token packing, etc.
        hidden_states = hidden_states[0]        # [batch_size, seq_len, hidden_dim]

        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Calculate routing logits for each token
        # router_logits: [batch_size, sequence_length, num_experts]
        cls_token_weights = self.gate(hidden_states[:, 0, :])  # [batch_size, num_experts]
        routing_weights = F.softmax(cls_token_weights, dim=-1)  # Softmax over num_experts

        # Find the top-1 expert for each image based on cls_token
        _, top_expert_idx = torch.max(cls_token_weights, dim=-1)  # [batch_size], each is the index of the top-1 expert

        # Prepare to gather the output of the chosen expert
        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # Loop through all images and apply their chosen experts
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            
            # For each image, apply the corresponding expert (based on top_expert_idx)
            expert_mask = (top_expert_idx == expert_idx)  # [batch_size], mask for the selected expert

            # Apply the expert to the tokens
            current_state = hidden_states
            current_hidden_states = expert_layer(current_state, hidden_states_multi)

            # Add the result only for the selected expert's images
            final_hidden_states[expert_mask] += current_hidden_states[expert_mask]

        # Now we need to handle padding, ensuring all images in the batch have the same number of tokens.
        # Get the number of remaining tokens (based on routing weights)
        remaining_tokens = routing_weights.sum(dim=-1)  # [batch_size, sequence_length], sum of routing weights for each token

        # Find the maximum remaining token length
        max_remaining_tokens = int(remaining_tokens.max().item())

        # Padding step: pad the final_hidden_states to match the maximum remaining tokens
        for i in range(batch_size):
            if final_hidden_states[i].shape[0] < max_remaining_tokens:
                pad_size = max_remaining_tokens - final_hidden_states[i].shape[0]
                final_hidden_states[i] = F.pad(final_hidden_states[i], (0, 0, 0, pad_size), value=0)

        return final_hidden_states