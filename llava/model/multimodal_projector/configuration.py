import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING

from transformers import Blip2QFormerConfig

logger = logging.get_logger(__name__)

class MplugOwlVisualAbstractorConfig(PretrainedConfig):
    model_type = "mplug_owl_visual_abstract"

    def __init__(
        self,
        add_v2t_pos_emb=False,
        use_cls_token=True,
        num_learnable_queries=64,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=2816,
        attention_probs_dropout_prob=0.,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        encoder_hidden_size=1024,
        grid_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_cls_token=use_cls_token
        self.add_v2t_pos_emb=add_v2t_pos_emb
        self.hidden_size = hidden_size
        self.num_learnable_queries = num_learnable_queries
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.grid_size = grid_size if grid_size else 32

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_abstractor config dict if we are loading from MplugOwlConfig
        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["abstractor_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class TokenPackerConfig():
    def __init__(
        self
    ):
        pass

class QFormerConfig(Blip2QFormerConfig):
    def __init__(self, num_learnable_queries, **kwargs):
        super().__init__(**kwargs)
        self.num_learnable_queries = num_learnable_queries


class MoEConfig():
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=11008,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.1,
        visual_abstractor=None,
        qformer_config=None,
        tokenpacker_config=None
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_jitter_noise = router_jitter_noise
        self.visual_abstractor_config = MplugOwlVisualAbstractorConfig(**visual_abstractor)
        self.qformer_config = QFormerConfig(**qformer_config)
        self.tokenpacker_config = TokenPackerConfig(**tokenpacker_config)