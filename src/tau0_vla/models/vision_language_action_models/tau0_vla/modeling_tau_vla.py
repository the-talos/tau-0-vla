"""Tau VLA: Qwen3.5 backbone + Qwen3.5 action expert with flow matching.

Architecture:
  - VLM backbone: Qwen3.5 (VL, frozen) with hybrid linear/full attention layers
  - Action Expert: Qwen3.5 text architecture, all layers = full_attention (no GatedDeltaNet)
  - Joint attention on VLM's full_attention layers
  - Independent processing on VLM's linear_attention layers (VLM: GatedDeltaNet, Expert: self-attention)

Key differences from LingBot VLA (Qwen2-based):
  - Gated Query Attention: q_proj -> [Q, gate], output = attn * sigmoid(gate)
  - QK Norm: built-in per-head RMSNorm
  - Partial RoPE: 25% of head_dim (64 out of 256)
  - RoPE applied BEFORE concat (VLM uses MRoPE, Expert uses 1D RoPE)
  - Qwen3.5 RMSNorm: (1 + weight) * norm(x), weight init to 0
  - head_dim=256, attention_bias=False
"""

import glob
import os
from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5MLP,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.utils import (
    ModelOutput,
    logging,
)

from tau0_vla.utils.profiling import sync_ts

from .qwen35vl_in_vla import (
    Qwen3_5_VLForConditionalGeneration,
    Qwen3_5_VLModel,
    Qwen3_5DecoderLayerForVLM,
)
from .utils import (
    causal_sdpa_flash_backend_forward,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    our_attention_forward,
    sample_beta,
)

logger = logging.get_logger(__name__)


# ===========================================================================
# Expert Attention (all layers are full_attention)
# ===========================================================================


class Qwen3_5AttentionForExpert(nn.Module):
    """Qwen3.5-style attention for the action expert.

    Same interface as VLM attention: compute_kqv / output_atten two-phase.
    Uses Gated Query Attention (q_proj -> [Q, gate]) and QK-Norm.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)


# ===========================================================================
# Expert Decoder Layer
# ===========================================================================


class Qwen3_5DecoderLayerForExpert(nn.Module):
    """Expert decoder layer. Always full_attention. Supports compute_kqv / output_atten
    and optional AdaNorm conditioning.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3_5AttentionForExpert(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: int = 0,
        end: int = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        ada_cond: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if compute_kqv:
            if ada_cond is not None:
                normed, gate_ada = self.input_layernorm(hidden_states, ada_cond)
            else:
                normed = self.input_layernorm(hidden_states)
                gate_ada = None

            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self.self_attn.head_dim)

            query_states, gate_q = torch.chunk(
                self.self_attn.q_proj(normed).view(*input_shape, -1, self.self_attn.head_dim * 2),
                2,
                dim=-1,
            )
            gate_q = gate_q.reshape(*input_shape, -1)

            query_states = self.self_attn.q_norm(query_states.view(hidden_shape))
            key_states = self.self_attn.k_norm(self.self_attn.k_proj(normed).view(hidden_shape))
            value_states = self.self_attn.v_proj(normed).view(hidden_shape)

            return query_states, key_states, value_states, gate_q

        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)

            # Apply sigmoid(gate) BEFORE o_proj (gate is in attention dim, not hidden dim)
            sliced = att_output[:, start:end]
            if gate is not None:
                sliced = sliced * torch.sigmoid(gate)
            out_emb = self.self_attn.o_proj(sliced)
            out_emb = out_emb + hidden_states

            # MLP
            residual = out_emb
            if ada_cond is not None:
                out_emb, _ = self.post_attention_layernorm(out_emb, ada_cond)
            else:
                out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb = residual + out_emb

            # Ensure output dtype matches layer parameters (softmax/sigmoid may upcast to float32)
            target_dtype = self.self_attn.o_proj.weight.dtype
            if out_emb.dtype != target_dtype:
                out_emb = out_emb.to(target_dtype)

            return out_emb

        else:
            raise ValueError(f"Invalid operation: compute_kqv={compute_kqv}, output_atten={output_atten}")


# ===========================================================================
# Expert Model
# ===========================================================================


class Qwen3_5ExpertModel(Qwen3_5PreTrainedModel):
    """Action expert backbone using Qwen3.5 architecture (all full_attention layers)."""

    config_class = Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayerForExpert(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class Qwen3_5ExpertForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Expert model wrapper with lm_head."""

    config_class = Qwen3_5TextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.model = Qwen3_5ExpertModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


# ===========================================================================
# AdaNorm for Expert
# ===========================================================================


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with FiLM conditioning, adapted for Qwen3.5 (1+weight) style."""

    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)
        self.gate = nn.Linear(cond_dim, hidden_size)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states, cond):
        output = self._norm(hidden_states.float())
        output = output * (1.0 + self.weight.float())
        output = output.type_as(hidden_states)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        gate = self.gate(cond).unsqueeze(1)
        output = (1 + gamma) * output + beta
        return output, gate


def replace_lnorm_with_adanorm(module, hidden_size, cond_dim):
    """Replace Qwen3_5RMSNorm layers with AdaRMSNorm in expert model."""
    for name, child in module.named_children():
        if isinstance(child, Qwen3_5RMSNorm) and "q_norm" not in name and "k_norm" not in name:
            setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_lnorm_with_adanorm(child, hidden_size, cond_dim)


# ===========================================================================
# Config: Qwen35vlWithExpert
# ===========================================================================


class Qwen35vlWithExpertConfig(PretrainedConfig):
    model_type = "Qwen35vlWithExpertModel"

    def __init__(
        self,
        qwenvl_config: dict | None = None,
        qwen_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        vocab_size: int = 248320,
        use_lm_head: bool = False,
        attention_implementation: str = "eager",
        adanorm_time: bool = False,
        tau_vla_prefix_flash_backend: bool = False,
        **kwargs,
    ):
        kwargs.pop("sdpa_flash_attn_mix", None)
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.adanorm_time = adanorm_time
        self.tau_vla_prefix_flash_backend = tau_vla_prefix_flash_backend

        # VLM config: must be provided as a dict (from VLM base config or saved checkpoint)
        self.qwenvl_config = qwenvl_config

        # Expert config: Qwen3.5 text config with ALL full_attention layers
        if qwen_expert_config is None:
            self.qwen_expert_config = Qwen3_5TextConfig(
                vocab_size=248320,
                hidden_size=768,
                intermediate_size=2752,
                num_hidden_layers=24,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=128,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=False,
                attention_bias=False,
                attention_dropout=0.0,
                # Force ALL layers to full_attention (no GatedDeltaNet)
                layer_types=["full_attention"] * 24,
                # Partial RoPE matching VLM
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                },
            )
        elif isinstance(qwen_expert_config, dict):
            self.qwen_expert_config = Qwen3_5TextConfig(**qwen_expert_config)
        else:
            self.qwen_expert_config = qwen_expert_config

        super().__init__(**kwargs)


# ===========================================================================
# Static KV Cache (for CUDA graph / torch.compile compatibility)
# ===========================================================================


class StaticKVCache:
    """Pre-allocated KV cache with fixed shapes for CUDA graph capture.

    Instead of the dict-based {layer_idx: {key_states, value_states}} which
    requires dynamic allocation and Python dict access (graph breaks), this
    stores KV tensors as contiguous pre-allocated buffers indexed by layer.

    Only ``full_attention`` layers keep a KV cache; ``linear_attention``
    (GatedDeltaNet) layers carry a recurrent state instead and never touch this
    cache. We therefore allocate a buffer only for full_attention layers and put
    ``None`` in the other slots — this saves memory while keeping ``key_cache`` a
    ``list`` indexed by the model's global ``layer_idx`` (unchanged container type,
    so the torch.compile / Dynamo treatment of the cache stays identical).
    """

    def __init__(self, layer_types, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype, device):
        shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
        self.key_cache = [
            torch.zeros(shape, dtype=dtype, device=device) if t == "full_attention" else None for t in layer_types
        ]
        self.value_cache = [
            torch.zeros(shape, dtype=dtype, device=device) if t == "full_attention" else None for t in layer_types
        ]
        self.seq_len = max_seq_len

    def fill(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        self.key_cache[layer_idx].copy_(key_states)
        self.value_cache[layer_idx].copy_(value_states)

    def get(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class DynamicKVCache:
    """Dict-backed growing KV cache for the eager / dynamic (max_prefix_len=0) path.

    Mirrors StaticKVCache's ``fill``/``get`` interface so ``handle_kv_cache`` can stay
    backend-agnostic. Unlike StaticKVCache it grows on demand and stores the exact
    prefix length (no padding, no pre-allocation).
    """

    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}

    def fill(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states

    def get(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# ===========================================================================
# Core Joint Model: VLM + Expert with joint/independent attention
# ===========================================================================


class Qwen35vlWithExpertModel(PreTrainedModel):
    config_class = Qwen35vlWithExpertConfig

    def __init__(self, config: Qwen35vlWithExpertConfig):
        super().__init__(config=config)
        self.config = config

        # Determine attention implementations for VLM standard forward vs VLA joint attention.
        # VLM standard forward (VQA / FAST-only) uses fully causal masks → FA2 is fine.
        # VLA joint attention uses non-causal 2D masks → FA2 cannot be used directly,
        # so we degrade to SDPA with explicit masks on the VLA path.
        user_attn_impl = self.config.attention_implementation
        if user_attn_impl == "flash_attention_2":
            # VLM standard path: keep FA2 if user requested it (fully causal, FA2 works)
            vlm_attn_impl = user_attn_impl
            # VLA joint path: force SDPA mode because it needs explicit 2D masks.
            self._vla_attn_mode = "sdpa"
            logger.warning_once(
                f"Tau VLA joint attention does not support pure flash_attention_2 "
                f"(non-causal 2D mask required). Degrading VLA attention to SDPA with explicit 2D masks. "
                f"VLM standard forward (VQA/FAST-only path) uses attn_implementation='{vlm_attn_impl}'."
            )
        else:
            vlm_attn_impl = user_attn_impl
            self._vla_attn_mode = user_attn_impl

        # Build VLM architecture from config dict (no weight loading — handled externally)
        vlm_config = AutoConfig.for_model(**self.config.qwenvl_config)
        # Propagate attention implementation to VLM config so its standard forward
        # (VQA / FAST-only path, fully causal) uses the appropriate attention kernel.
        vlm_config._attn_implementation = vlm_attn_impl
        if hasattr(vlm_config, "text_config"):
            vlm_config.text_config._attn_implementation = vlm_attn_impl
        self.qwenvl = Qwen3_5_VLForConditionalGeneration._from_config(vlm_config)

        # Ensure expert matches VLM on critical dimensions for joint attention
        vlm_text_config = vlm_config.text_config if hasattr(vlm_config, "text_config") else vlm_config
        vlm_num_layers = vlm_text_config.num_hidden_layers
        if self.config.qwen_expert_config.num_hidden_layers != vlm_num_layers:
            self.config.qwen_expert_config.num_hidden_layers = vlm_num_layers
            self.config.qwen_expert_config.layer_types = ["full_attention"] * vlm_num_layers

        # Force expert head counts and head_dim to match VLM (required for joint attention concat on seq dim)
        self.config.qwen_expert_config.num_attention_heads = vlm_text_config.num_attention_heads
        self.config.qwen_expert_config.num_key_value_heads = vlm_text_config.num_key_value_heads
        self.config.qwen_expert_config.head_dim = vlm_text_config.head_dim

        # Build expert
        self.qwen_expert = Qwen3_5ExpertForCausalLM(self.config.qwen_expert_config)

        # Optional AdaNorm for timestep conditioning
        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
            )

        # Remove unused embed_tokens and patch get_input_embeddings to return None
        # so that enable_input_require_grads() (gradient checkpointing) skips it.
        del self.qwen_expert.model.embed_tokens
        self.qwen_expert.model.get_input_embeddings = lambda: None
        self.qwen_expert.get_input_embeddings = lambda: None

        self._vla_attn_mode = self.config.attention_implementation
        self.attention_interface = self.get_attention_interface()
        self.set_requires_grad()

        # Store VLM layer_types for the forward loop
        self._vlm_layer_types = vlm_text_config.layer_types

        # Fix: fla's chunk_gated_delta_rule CUDA kernel produces NaN at chunk
        # boundaries (token 256+) with extreme layernorm output values.
        # The torch fallback is numerically stable (casts to float32 internally).
        # self._patch_gdn_use_torch_fallback()

    def _patch_gdn_use_torch_fallback(self):
        """Replace fla CUDA kernels with torch fallbacks in all GatedDeltaNet layers.

        fla's chunk_gated_delta_rule CUDA kernel produces NaN at chunk boundaries
        (around token 256) when layernorm output has extreme values (e.g. [-55, 11]).
        This happens because the VLA concatenates image + text embeddings with very
        different scales, creating a distribution the kernel wasn't designed for.
        The torch fallback casts to float32 internally and is numerically stable.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        vlm_text = self.qwenvl.language_model
        for layer in vlm_text.layers:
            if hasattr(layer, "linear_attn"):
                gdn = layer.linear_attn
                gdn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                gdn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
            for params in self.qwenvl.visual.parameters():
                params.requires_grad = False
        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()
        return self

    def get_image_features(self, pixel_values, image_grid_thw=None):
        """Returns stacked [num_images, tokens_per_image, D].

        Requires all images in the batch to share the same token count
        (same H*W/merge²). Action-path data satisfies this by construction
        (fixed robot-camera resolution). For variable-size inputs use
        get_video_features, which returns a flat [total_tokens, D] tensor.
        """
        image_embeds = self.qwenvl.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
        # image_embeds is a tuple of tensors (split by image)
        image_embeds = torch.stack(image_embeds, dim=0)
        return image_embeds

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        """Returns flat [total_video_tokens, D]; no stacking because videos may
        have varying T/H/W across the batch."""
        return self.qwenvl.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.language_model.embed_tokens(tokens)

    def handle_kv_cache(
        self, key_states, value_states, layer_idx, past_key_values=None, use_cache=None, fill_kv_cache=None
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicKVCache()
            if fill_kv_cache:
                if hasattr(past_key_values, "fill"):
                    past_key_values.fill(layer_idx, key_states, value_states)
                else:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
            else:
                if isinstance(past_key_values, dict):
                    cached_k = past_key_values[layer_idx]["key_states"]
                    cached_v = past_key_values[layer_idx]["value_states"]
                else:
                    cached_k, cached_v = past_key_values.get(layer_idx)
                key_states = torch.cat([cached_k, key_states], dim=1)
                value_states = torch.cat([cached_v, value_states], dim=1)
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: list = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        ada_cond=None,
        vlm_pad_mask: Optional[torch.Tensor] = None,
        use_prefix_flash_backend: bool = False,
    ):
        """Forward pass with joint attention on full_attention layers,
        independent processing on linear_attention layers.

        Args:
            inputs_embeds: [vlm_hidden, expert_hidden] where either can be None
            position_ids: position ids for the FULL concatenated sequence (vlm + expert)
                          Shape: [B, total_seq_len] for 1D RoPE
            vlm_position_ids: 3D MRoPE position ids for VLM tokens
                              Shape: [3, B, vlm_seq_len] for MRoPE
            vlm_pad_mask: 2D padding mask for VLM tokens [B, vlm_seq_len],
                          1=valid 0=padding. Used by GatedDeltaNet linear_attention
                          layers to zero out padding hidden states.
            use_prefix_flash_backend: If True, full_attention VLM-prefix
                self-attention is routed through SDPA's Flash backend
                (``is_causal=True``, no explicit mask). Only valid for
                right-padded causal prefix masks; suffix/action attention
                still uses explicit masks. Right-padding is guaranteed by
                the data collator contract (verified once on the first
                batch in ``DataCollatorForVLMSupervisedDataset``), so no
                runtime check is performed here.
        """
        vlm_model = self.qwenvl.language_model  # Qwen3_5VLMTextModel
        expert_model = self.qwen_expert.model  # Qwen3_5ExpertModel

        num_layers = vlm_model.config.num_hidden_layers  # 24 for 2b
        vlm_hidden = inputs_embeds[0]
        expert_hidden = inputs_embeds[1]
        if attention_mask is None:
            raise ValueError(
                "attention_mask is required for Tau0VLA joint attention so prefix, suffix, "
                "and padding masking semantics are explicit."
            )
        mask_2d = attention_mask.bool()
        use_prefix_flash_backend = bool(
            use_prefix_flash_backend and self._vla_attn_mode == "sdpa" and vlm_pad_mask is not None
        )

        # Compute position embeddings for VLM (MRoPE) and Expert (1D RoPE)
        # VLM uses 3D position ids (temporal, height, width)
        # Expert uses 1D position ids
        if vlm_hidden is not None:
            # VLM MRoPE: vlm_position_ids shape [3, B, vlm_seq_len]
            if vlm_position_ids is not None:
                if vlm_position_ids.ndim == 2:
                    vlm_position_ids_3d = vlm_position_ids[None, ...].expand(3, vlm_position_ids.shape[0], -1)
                elif vlm_position_ids.ndim == 3 and vlm_position_ids.shape[0] == 3:
                    vlm_position_ids_3d = vlm_position_ids
                else:
                    vlm_position_ids_3d = vlm_position_ids[None, ...].expand(3, vlm_position_ids.shape[0], -1)
            else:
                vlm_seq_len = vlm_hidden.shape[1]
                vlm_position_ids_3d = torch.arange(vlm_seq_len, device=vlm_hidden.device)[None, None, :].expand(
                    3, vlm_hidden.shape[0], -1
                )
            vlm_cos, vlm_sin = vlm_model.rotary_emb(vlm_hidden, vlm_position_ids_3d)

        if expert_hidden is not None:
            # Expert 1D RoPE: extract expert portion from position_ids
            if vlm_hidden is not None:
                vlm_seq_len = vlm_hidden.shape[1]
                exp_position_ids_1d = position_ids[:, vlm_seq_len:]  # [B, exp_seq_len]
            else:
                exp_position_ids_1d = position_ids  # [B, exp_seq_len]

            # Expert uses 1D RoPE, but rotary_emb expects 3D (MRoPE format)
            exp_position_ids_3d = exp_position_ids_1d[None, ...].expand(3, -1, -1)
            exp_cos, exp_sin = expert_model.rotary_emb(expert_hidden, exp_position_ids_3d)

        for layer_idx in range(num_layers):
            vlm_layer = vlm_model.layers[layer_idx]
            expert_layer = expert_model.layers[layer_idx]

            vlm_layer_type = self._vlm_layer_types[layer_idx]

            if vlm_layer_type == "full_attention":
                # === JOINT ATTENTION ===

                # Phase 1: compute Q, K, V from both models
                if vlm_hidden is not None:  # (16, 318, 2048)
                    vlm_q, vlm_k, vlm_v, vlm_gate = vlm_layer(vlm_hidden, compute_kqv=True)

                if expert_hidden is not None:
                    exp_q, exp_k, exp_v, exp_gate = expert_layer(expert_hidden, compute_kqv=True, ada_cond=ada_cond)

                # Phase 2: Apply RoPE BEFORE concatenation
                # VLM: MRoPE (3D positional encoding)
                if vlm_hidden is not None:
                    # vlm_q: [B, L, H, D], need [B, H, L, D] for apply_rotary_pos_emb
                    vlm_q_t = vlm_q.transpose(1, 2)
                    vlm_k_t = vlm_k.transpose(1, 2)
                    vlm_q_t, vlm_k_t = apply_rotary_pos_emb(vlm_q_t, vlm_k_t, vlm_cos, vlm_sin)
                    vlm_q = vlm_q_t.transpose(1, 2)  # back to [B, L, H, D]
                    vlm_k = vlm_k_t.transpose(1, 2)

                # Expert: 1D RoPE
                if expert_hidden is not None:
                    exp_q_t = exp_q.transpose(1, 2)
                    exp_k_t = exp_k.transpose(1, 2)
                    exp_q_t, exp_k_t = apply_rotary_pos_emb(exp_q_t, exp_k_t, exp_cos, exp_sin)
                    exp_q = exp_q_t.transpose(1, 2)
                    exp_k = exp_k_t.transpose(1, 2)

                # Phase 3: Concatenate and do joint attention
                query_parts = []
                key_parts = []
                value_parts = []
                gates = []

                if vlm_hidden is not None:
                    query_parts.append(vlm_q)
                    key_parts.append(vlm_k)
                    value_parts.append(vlm_v)
                    gates.append(vlm_gate)

                if expert_hidden is not None:
                    query_parts.append(exp_q)
                    key_parts.append(exp_k)
                    value_parts.append(exp_v)
                    gates.append(exp_gate)

                key_states = torch.cat(key_parts, dim=1)
                value_states = torch.cat(value_parts, dim=1)

                key_states, value_states, past_key_values = self.handle_kv_cache(
                    key_states,
                    value_states,
                    layer_idx,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                )

                # Phase 3: Attention
                # VLA action path has two mask semantics in the same layer:
                #   VLM prefix: explicit prefix mask (causal when vlm_causal=True).
                #   Expert suffix: explicit block mask so action tokens can attend
                #                  bidirectionally within the action chunk.
                # Both paths use explicit masks here.
                if vlm_hidden is not None and expert_hidden is not None:
                    vlm_seq_len = vlm_hidden.shape[1]
                    mask_2d_exp = mask_2d[:, vlm_seq_len:, :]

                    if use_prefix_flash_backend:
                        vlm_att = causal_sdpa_flash_backend_forward(vlm_q, vlm_k, vlm_v)
                    else:
                        vlm_att = self.attention_interface(
                            vlm_q,
                            vlm_k,
                            vlm_v,
                            mask_2d[:, :vlm_seq_len, :vlm_seq_len],
                        )

                    exp_key = torch.cat([vlm_k, exp_k], dim=1)
                    exp_val = torch.cat([vlm_v, exp_v], dim=1)
                    exp_att = self.attention_interface(
                        exp_q,
                        exp_key,
                        exp_val,
                        mask_2d_exp,
                    )

                    att_output = torch.cat([vlm_att, exp_att], dim=1)
                else:
                    query_states = torch.cat(query_parts, dim=1)
                    if (
                        use_prefix_flash_backend
                        and vlm_hidden is not None
                        and expert_hidden is None
                        and key_states.shape[1] == query_states.shape[1]
                    ):
                        att_output = causal_sdpa_flash_backend_forward(vlm_q, key_states, value_states)
                    else:
                        att_output = self.attention_interface(
                            query_states,
                            key_states,
                            value_states,
                            mask_2d,
                        )

                # Phase 4: Split and apply output (gate, residual, MLP)
                outputs_embeds = []
                start = 0
                for i, hidden in enumerate(inputs_embeds):
                    if hidden is not None:
                        end = start + hidden.shape[1]
                        if i == 0:  # VLM
                            out_emb = vlm_layer(
                                vlm_hidden,
                                att_output,
                                start,
                                end,
                                output_atten=True,
                                gate=gates[0] if gates else None,
                            )
                        else:  # Expert
                            gate_idx = 0 if len(gates) == 1 else i
                            out_emb = expert_layer(
                                expert_hidden,
                                att_output,
                                start,
                                end,
                                output_atten=True,
                                ada_cond=ada_cond,
                                gate=gates[gate_idx],
                            )
                        outputs_embeds.append(out_emb)
                        start = end
                    else:
                        outputs_embeds.append(None)

                vlm_hidden = outputs_embeds[0]
                expert_hidden = outputs_embeds[1]

            elif vlm_layer_type == "linear_attention":
                # === INDEPENDENT PROCESSING ===

                # VLM: GatedDeltaNet (standard forward)
                if vlm_hidden is not None:
                    # Compute linear attention mask following transformers convention:
                    # pass 2D padding mask [B, L] so GatedDeltaNet zeros padding states.
                    # NOTE: no `torch.all(mask == 1)` short-circuit here — that is a
                    # data-dependent host sync that breaks CUDA-graph capture / torch.compile
                    # on the static-prefix (optim) path.
                    linear_attn_mask = vlm_pad_mask
                    vlm_hidden = vlm_layer(
                        vlm_hidden,
                        attention_mask=linear_attn_mask,
                        past_key_values=None,
                    )

                # Expert: self-attention over suffix tokens. Action tokens use
                # an explicit block mask, not the causal flash-attention path.
                if expert_hidden is not None:
                    exp_q, exp_k, exp_v, exp_gate = expert_layer(expert_hidden, compute_kqv=True, ada_cond=ada_cond)

                    # Apply RoPE
                    exp_q_t = exp_q.transpose(1, 2)  # (bs, 31, 8, 256) (bs, n, h, d)
                    exp_k_t = exp_k.transpose(1, 2)  # (bs, 31, 2, 256) (bs, n, h, d)
                    exp_q_t, exp_k_t = apply_rotary_pos_emb(exp_q_t, exp_k_t, exp_cos, exp_sin)
                    exp_q = exp_q_t.transpose(1, 2)
                    exp_k = exp_k_t.transpose(1, 2)

                    exp_seq_len = expert_hidden.shape[1]
                    if vlm_hidden is not None:
                        vlm_seq_len = inputs_embeds[0].shape[1] if inputs_embeds[0] is not None else 0
                    else:
                        vlm_seq_len = 0
                    expert_mask = mask_2d[:, vlm_seq_len:, -exp_seq_len:]

                    exp_att = self.attention_interface(exp_q, exp_k, exp_v, expert_mask)

                    expert_hidden = expert_layer(
                        expert_hidden,
                        exp_att,
                        0,
                        exp_seq_len,
                        output_atten=True,
                        ada_cond=ada_cond,
                        gate=exp_gate,
                    )

        # Final norm
        outputs_embeds = []
        for i, hidden in enumerate([vlm_hidden, expert_hidden]):
            if hidden is not None:
                if i == 0:
                    out_emb = vlm_model.norm(hidden)
                else:
                    if ada_cond is not None and hasattr(expert_model.norm, "gamma"):
                        out_emb, _ = expert_model.norm(hidden, ada_cond)
                    else:
                        out_emb = expert_model.norm(hidden)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        # Use the VLA-specific attention mode (may differ from config.attention_implementation
        # when FA2 is requested — VLA degrades to SDPA+FA2 mix due to non-causal masks).
        attn_mode = self._vla_attn_mode
        if attn_mode not in ("eager", "sdpa", "flash_attention_2"):
            raise ValueError(
                f"Invalid attention implementation: {attn_mode}. Expected 'eager', 'sdpa', or 'flash_attention_2'."
            )
        from functools import partial

        return partial(our_attention_forward, mode=attn_mode)


# ===========================================================================
# Tau0VLA Config
# ===========================================================================


class Tau0VLAConfig(PretrainedConfig):
    """Configuration for Tau VLA (Flow-Matching VLA with Qwen3.5 + Qwen3.5 action expert).

    Note on removed keys: released checkpoints carry training-only flags that no
    longer exist here (notably ``detach_vlm_for_vla_loss``, the knowledge-insulation
    switch). ``PretrainedConfig.__init__`` stores unrecognized ``config.json`` keys
    as plain attributes via ``**kwargs``, so such checkpoints still load; the flags
    are simply inert. They only ever gated gradient flow during training and have no
    effect at inference. Action training is the only shipped route.
    """

    model_type = "tau_vla"

    def __init__(
        self,
        qwenvl_config=None,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=False,
        vocab_size=248320,
        use_lm_head=False,
        attention_implementation="eager",
        # Action head
        max_action_dim=32,
        max_state_dim=32,
        action_dim=16,
        n_action_steps=50,
        proj_width=768,
        loss_type="fm",
        num_steps=10,
        use_cache=True,
        # AdaNorm
        adanorm_time=False,
        # Prefix mask: True=causal (aligned with Qwen3.5 pretraining), False=bidirectional
        zero_state_emb=False,
        vlm_causal=True,
        tau_vla_prefix_flash_backend=False,
        action_rope_offset=0,
        use_action_mask_loss=True,
        max_prefix_len=0,  # 0 = disable padding (dynamic shapes, no compile optimization)
        **kwargs,
    ):
        kwargs.pop("sdpa_flash_attn_mix", None)
        self.qwenvl_config = qwenvl_config
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_state_proj = train_state_proj
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.attention_implementation = attention_implementation
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.proj_width = proj_width
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.adanorm_time = adanorm_time
        self.vlm_causal = vlm_causal
        self.tau_vla_prefix_flash_backend = tau_vla_prefix_flash_backend
        self.zero_state_emb = zero_state_emb
        self.action_rope_offset = action_rope_offset
        self.use_action_mask_loss = use_action_mask_loss
        self.max_prefix_len = max_prefix_len
        # TODO: remove this parameter once all legacy checkpoints are migrated
        if action_rope_offset != 0:
            logger.warning(
                f"action_rope_offset={action_rope_offset} is non-zero. "
                "This should only be used for compatibility with old pretrained checkpoints "
                "that were trained with the buggy position-id counting. "
                "For new training runs, set action_rope_offset=0."
            )
        super().__init__(**kwargs)


# ===========================================================================
# Tau0VLA Output
# ===========================================================================


@dataclass
class Tau0VLAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    vla_loss: Optional[torch.FloatTensor] = None


# ===========================================================================
# Flow Matching
# ===========================================================================


class FlowMatching(nn.Module):
    def __init__(self, config: Tau0VLAConfig):
        super().__init__()
        self.config = config

        qwenvl_with_expert_config = Qwen35vlWithExpertConfig(
            qwenvl_config=self.config.qwenvl_config,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            adanorm_time=getattr(config, "adanorm_time", False),
            tau_vla_prefix_flash_backend=getattr(config, "tau_vla_prefix_flash_backend", False),
        )
        self.qwenvl_with_expert = Qwen35vlWithExpertModel(qwenvl_with_expert_config)
        self.config.proj_width = qwenvl_with_expert_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(qwenvl_with_expert_config.qwen_expert_config, "initializer_range", 0.02)

        # Projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        # Pre-computed constants for the denoising path
        self._use_adanorm = getattr(config, "adanorm_time", False)
        self._n_action_steps = config.n_action_steps

        # CUDA graph state for prefix (lazy-initialized on first inference)
        self._cuda_graph_prefix = None
        self._cuda_graph_prefix_signature = None
        # torch.compile state for denoising (lazy-initialized on first inference)
        self._compiled_predict_velocity = None
        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _allow_prefix_flash_backend(self) -> bool:
        return bool(getattr(self.config, "tau_vla_prefix_flash_backend", False)) and self.config.vlm_causal

    @staticmethod
    def _suffix_masks(batch_size: int, suffix_len: int, device):
        """Denoise suffix mask pattern: every token valid (pad), first 2 causal (att).

        Single source of truth shared by ``embed_suffix`` (eager path) and
        ``_build_denoise_masks`` (static/compiled path) — they MUST produce identical
        suffix masks, so the pattern lives here once.
        """
        pad_masks = torch.ones((batch_size, suffix_len), device=device, dtype=torch.bool)
        att_masks = torch.zeros((batch_size, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, :2] = True
        return pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        state_emb = self.state_proj(state)
        if self.config.zero_state_emb:
            state_emb = torch.zeros_like(state_emb)

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb_ori = time_emb

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks, att_masks = self._suffix_masks(bsize, action_time_dim + 1, device)

        return time_emb_ori, embs, pad_masks, att_masks

    def _build_action_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        timing: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prefix embeddings and MRoPE position_ids for the action path.

        Uses the same embed-then-replace approach as Qwen3.5's standard VQA
        forward (``masked_scatter``), ensuring ``<|vision_start|>`` /
        ``<|vision_end|>`` tokens are preserved and token order matches VLM
        pretraining exactly.

        Returns:
            prefix_embs:      [B, L, D] — input embeddings with ``<|image_pad|>``
                               replaced by vision features.
            prefix_pad_masks: [B, L]    — attention mask (1=valid, 0=pad).
            vlm_position_ids: [3, B, L] — 3D MRoPE position_ids.
        """
        _t = timing is not None
        qwenvl = self.qwenvl_with_expert.qwenvl
        image_token_id = getattr(qwenvl.config, "image_token_id", None)
        video_token_id = getattr(qwenvl.config, "video_token_id", None)

        # 1. Embed all tokens (including image_pad/video_pad, vision_start, vision_end)
        t0 = sync_ts(_t)
        prefix_embs = self.qwenvl_with_expert.embed_language_tokens(input_ids)
        t1 = sync_ts(_t)

        # 2. Replace <|image_pad|> placeholders with vision features
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = qwenvl.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)  # [total_vis_tokens, D]
            image_mask = input_ids == image_token_id
            prefix_embs[image_mask] = image_embeds.to(prefix_embs.dtype)
        t2 = sync_ts(_t)

        # 3. Replace <|video_pad|> placeholders with video features
        if pixel_values_videos is not None and video_grid_thw is not None:
            # get_video_features returns flat [total_video_tokens, D] (no split/tuple)
            video_embeds = qwenvl.model.get_video_features(pixel_values_videos, video_grid_thw)
            video_mask = input_ids == video_token_id
            prefix_embs[video_mask] = video_embeds.to(prefix_embs.dtype)
        t3 = sync_ts(_t)

        # 4. Build 3D MRoPE position_ids via get_rope_index (same as VQA path)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        vlm_position_ids, _ = qwenvl.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        t4 = sync_ts(_t)

        if _t:
            timing["token_embed_ms"] = (t1 - t0) * 1000
            if pixel_values is not None and image_grid_thw is not None:
                timing["vision_encode_ms"] = (t2 - t1) * 1000
            if pixel_values_videos is not None and video_grid_thw is not None:
                timing["video_encode_ms"] = (t3 - t2) * 1000
            timing["rope_index_ms"] = (t4 - t3) * 1000

        return prefix_embs, attention_mask, vlm_position_ids

    def _prefix_forward_core(
        self,
        prefix_embs,
        prefix_att_2d_masks,
        prefix_position_ids,
        vlm_position_ids,
        kv_cache,
        vlm_pad_mask,
        use_prefix_flash_backend,
    ):
        """Prefix forward (no timing/getattr). Single definition shared by the eager,
        warmup, and CUDA-graph-capture paths; the ``kv_cache`` arg selects which (Dynamic
        vs StaticKVCache).

        Extracted mainly so warmup and capture run byte-identical code — otherwise two
        out-of-sync copies could make capture record different ops than were warmed up
        (autotuning/alloc leaking into the graph). Also serves as a clean capture / compile target.
        """
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=kv_cache,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            vlm_pad_mask=vlm_pad_mask,
            use_prefix_flash_backend=use_prefix_flash_backend,
        )
        return past_key_values

    def forward(
        self,
        state,
        actions,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        noise=None,
        time=None,
        loss_type="fm",
        pixel_values_videos=None,
        video_grid_thw=None,
        action_mask=None,
    ) -> Tensor:
        # Cast float inputs to model dtype (e.g. bf16) for FA2 compatibility
        param_dtype = next(self.parameters()).dtype
        state = state.to(dtype=param_dtype)
        actions = actions.to(dtype=param_dtype)

        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)

        # Scheme C (config.vla_inactive_input_zero): zero the noise on inactive
        # action dims so x_t = t*noise + (1-t)*action is EXACTLY 0 there (the
        # dataloader already zeroes action on inactive dims). The constant-0
        # pattern is an explicit per-sample mask signal to the network,
        # replacing scheme A's t*noise nuisance input, and matches the per-step
        # re-zeroing in sample_actions. u_t on those dims becomes 0 and is
        # excluded by the loss mask anyway.
        if action_mask is not None:
            am = action_mask.to(device=device, dtype=dtype).reshape(actions.shape[0], 1, -1)
            if am.shape[-1] < actions.shape[-1]:
                am = torch.nn.functional.pad(am, (0, actions.shape[-1] - am.shape[-1]))
            noise = noise * am
        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, vlm_position_ids = self._build_action_prefix(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )
        prefix_embs = prefix_embs.to(dtype=param_dtype)
        B, L = prefix_embs.shape[:2]
        if not self.config.vlm_causal:
            prefix_att_masks = torch.zeros(B, L, device=device, dtype=torch.bool)
        else:
            prefix_att_masks = torch.ones(B, L, device=device, dtype=torch.bool)

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        action_rope_offset = getattr(self.config, "action_rope_offset", 0)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1 + action_rope_offset

        (_, suffix_out), _ = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=True,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            vlm_pad_mask=prefix_pad_masks,
            use_prefix_flash_backend=self._allow_prefix_flash_backend(),
        )

        # --- Flow matching loss (unchanged) ---
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if suffix_out.dtype != self.action_out_proj.weight.dtype:
            suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        if loss_type == "fm":
            fm_losses = F.mse_loss(u_t, v_t, reduction="none")
        elif loss_type == "L1_fm":
            fm_losses = F.l1_loss(u_t, v_t, reduction="none")

        return fm_losses

    def sample_actions(
        self,
        state,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        noise=None,
        action_mask=None,
        timing: Optional[dict] = None,
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
        _t = timing is not None

        # noise initialization
        t0 = sync_ts(_t)
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = torch.randn(actions_shape, device=device, dtype=dtype)
        t1 = sync_ts(_t)

        # Inactive-dim train/inference consistency fix. Training only ever feeds
        # the network x_t = t*noise on inactive dims (action=0 there, FM loss
        # masked), but free Euler integration lets those dims drift OOD through
        # the unsupervised velocity components and leak into the shared
        # action_in_proj embedding. With the per-sample ``action_mask`` we pin
        # them back to the training input distribution at EVERY step:
        #   default                      -> inactive := t * noise   (scheme A)
        #   config.vla_inactive_input_zero -> inactive := 0         (scheme C)
        # action_mask=None keeps the legacy free behaviour (old callers).
        inactive_mask = None
        if action_mask is not None:
            inactive_mask = torch.zeros(bsize, 1, self.config.max_action_dim, device=device, dtype=dtype)
            am = action_mask.to(device=device, dtype=dtype).reshape(bsize, -1)
            inactive_mask[:, 0, : am.shape[-1]] = am
        zero_inactive = bool(getattr(self.config, "vla_inactive_input_zero", False))

        # build prefix — token embed + vision encode + RoPE ids + static masks
        build_prefix_timing: dict = {} if _t else None
        t_prefix = sync_ts(_t)
        prefix_embs, prefix_pad_masks, vlm_position_ids = self._build_action_prefix(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            timing=build_prefix_timing,
        )
        t_pad = sync_ts(_t)
        B, L = prefix_embs.shape[:2]

        # Right-pad prefix to max_prefix_len for static shapes (CUDA graph compatibility)
        max_prefix_len = self.config.max_prefix_len
        if _t:
            build_prefix_timing["prefix_len"] = int(L)
            build_prefix_timing["max_prefix_len"] = int(max_prefix_len)
        if max_prefix_len > 0 and L > max_prefix_len:
            raise AssertionError(
                f"Tau0VLA prefix length {L} exceeds max_prefix_len={max_prefix_len}. "
                "Static-prefix deploy optim requires every real input to fit the padded prefix budget; "
                "increase --max-prefix-len or use --infer-mode eager."
            )
        static_requested = max_prefix_len > 0 and L <= max_prefix_len
        # CUDA graph capture is CUDA-only (torch.cuda.CUDAGraph); on XPU and CPU the
        # dynamic eager path below is used, so say so once instead of silently skipping
        # the optimisation the caller asked for.
        use_static_cache = static_requested and device.type == "cuda" and torch.cuda.is_available()
        if static_requested and not use_static_cache:
            logger.warning_once(
                "max_prefix_len=%s requests the static prefix/CUDA-graph path, which is CUDA-only; "
                "falling back to the dynamic eager path on device type '%s'.",
                max_prefix_len,
                device.type,
            )
        if use_static_cache and L < max_prefix_len:
            pad_len = max_prefix_len - L
            prefix_embs = F.pad(prefix_embs, (0, 0, 0, pad_len))  # [B, max_prefix_len, D]
            prefix_pad_masks = F.pad(prefix_pad_masks.long(), (0, pad_len), value=0).bool()
            vlm_position_ids = F.pad(vlm_position_ids, (0, pad_len), value=0)  # [3, B, max_prefix_len]
            L = max_prefix_len

        t_mask = sync_ts(_t)
        no_prefix_padding = bool(torch.all(prefix_pad_masks == 1))
        use_prefix_flash_backend = bool(no_prefix_padding and self._allow_prefix_flash_backend())
        if not self.config.vlm_causal:
            prefix_att_masks = torch.zeros(B, L, device=device, dtype=torch.bool)
        else:
            prefix_att_masks = torch.ones(B, L, device=device, dtype=torch.bool)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        t2 = sync_ts(_t)
        if _t:
            build_prefix_timing["action_prefix_ms"] = (t_pad - t_prefix) * 1000
            build_prefix_timing["static_padding_ms"] = (t_mask - t_pad) * 1000
            build_prefix_timing["att_mask_position_build_ms"] = (t2 - t_mask) * 1000

        # VLM prefix forward — store KV cache for denoising reuse
        t3 = sync_ts(_t)
        vlm_pad_mask = prefix_pad_masks

        # Use StaticKVCache for fixed-shape operation when max_prefix_len is set.
        # The cache is part of the captured graph state, so allocate it only once.
        if use_static_cache:
            graph_signature = (
                B,
                L,
                dtype,
                device,
                bool(use_prefix_flash_backend),
                prefix_att_2d_masks is None,
                vlm_pad_mask is None,
            )
            if graph_signature != self._cuda_graph_prefix_signature:
                self._cuda_graph_prefix = None
                self._cuda_graph_prefix_signature = graph_signature
                self._compiled_predict_velocity = None
            if self._cuda_graph_prefix is None:
                # First call: allocate static buffers and capture CUDA graph
                vlm_text_cfg = self.qwenvl_with_expert.qwenvl.config.text_config
                static_cache = StaticKVCache(
                    layer_types=vlm_text_cfg.layer_types,
                    batch_size=B,
                    max_seq_len=L,
                    num_kv_heads=vlm_text_cfg.num_key_value_heads,
                    head_dim=vlm_text_cfg.head_dim,
                    dtype=dtype,
                    device=device,
                )
                self._graph_prefix_buffers = {
                    "prefix_embs": prefix_embs.clone(),
                    "prefix_att_2d_masks": prefix_att_2d_masks.clone() if prefix_att_2d_masks is not None else None,
                    "prefix_position_ids": prefix_position_ids.clone(),
                    "vlm_position_ids": vlm_position_ids.clone(),
                    "vlm_pad_mask": vlm_pad_mask.clone() if vlm_pad_mask is not None else None,
                    "static_cache": static_cache,
                }
                # CUDA-graph capture protocol warmup (not the openloop eval warmup): a side
                # stream is required so lazy init (cuBLAS/cuDNN autotune, allocator) finishes
                # before capture. 3 iters is the PyTorch convention. Replay later runs on the
                # main stream — fine, a graph is stream-agnostic with fixed buffer addresses.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._prefix_forward_core(
                            self._graph_prefix_buffers["prefix_embs"],
                            self._graph_prefix_buffers["prefix_att_2d_masks"],
                            self._graph_prefix_buffers["prefix_position_ids"],
                            self._graph_prefix_buffers["vlm_position_ids"],
                            self._graph_prefix_buffers["static_cache"],
                            self._graph_prefix_buffers["vlm_pad_mask"],
                            use_prefix_flash_backend,
                        )
                torch.cuda.current_stream().wait_stream(s)

                # Capture
                self._cuda_graph_prefix = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._cuda_graph_prefix):
                    self._prefix_forward_core(
                        self._graph_prefix_buffers["prefix_embs"],
                        self._graph_prefix_buffers["prefix_att_2d_masks"],
                        self._graph_prefix_buffers["prefix_position_ids"],
                        self._graph_prefix_buffers["vlm_position_ids"],
                        self._graph_prefix_buffers["static_cache"],
                        self._graph_prefix_buffers["vlm_pad_mask"],
                        use_prefix_flash_backend,
                    )

            # Replay: copy inputs into static buffers, replay graph
            self._graph_prefix_buffers["prefix_embs"].copy_(prefix_embs)
            if prefix_att_2d_masks is not None:
                self._graph_prefix_buffers["prefix_att_2d_masks"].copy_(prefix_att_2d_masks)
            self._graph_prefix_buffers["prefix_position_ids"].copy_(prefix_position_ids)
            self._graph_prefix_buffers["vlm_position_ids"].copy_(vlm_position_ids)
            if vlm_pad_mask is not None:
                self._graph_prefix_buffers["vlm_pad_mask"].copy_(vlm_pad_mask)
            self._cuda_graph_prefix.replay()
            past_key_values = self._graph_prefix_buffers["static_cache"]
        else:
            past_key_values = self._prefix_forward_core(
                prefix_embs,
                prefix_att_2d_masks,
                prefix_position_ids,
                vlm_position_ids,
                DynamicKVCache(),
                vlm_pad_mask,
                use_prefix_flash_backend,
            )
        t4 = sync_ts(_t)

        # Flow-matching denoising loop
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        t_loop = sync_ts(_t)
        num_steps = self.config.num_steps
        denoise_per_step = None

        if use_static_cache:
            suffix_len = self._n_action_steps + 1
            full_att_2d_masks, position_ids = self._build_denoise_masks(prefix_pad_masks, suffix_len)

            # Static-shape optimized path: prefix CUDA graph + denoising torch.compile.
            if self._compiled_predict_velocity is None:
                self._compiled_predict_velocity = torch.compile(self._predict_velocity_core, mode="reduce-overhead")

            for step_idx in range(num_steps):
                time_val = time - step_idx / num_steps
                # Re-pin inactive dims each step (same schedule as the eager loop below).
                if inactive_mask is not None:
                    if zero_inactive:
                        x_t = x_t * inactive_mask
                    else:
                        x_t = x_t * inactive_mask + (time_val * noise) * (1 - inactive_mask)
                timestep = time_val.expand(bsize)
                v_t = self._compiled_predict_velocity(
                    state, x_t, timestep, past_key_values, full_att_2d_masks, position_ids
                )
                x_t = x_t + dt * v_t
        else:
            # Fallback path: dynamic shapes (max_prefix_len not set) — the original eager loop.
            denoise_per_step = [] if _t else None
            step_idx = 0
            while time >= -dt / 2:
                if inactive_mask is not None:
                    if zero_inactive:
                        x_t = x_t * inactive_mask
                    else:
                        x_t = x_t * inactive_mask + (time * noise) * (1 - inactive_mask)
                expanded_time = time.expand(bsize)
                step_timing: dict = {} if _t else None
                v_t = self.predict_velocity(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time, timing=step_timing
                )
                x_t = x_t + dt * v_t
                time += dt
                if _t:
                    step_timing["step"] = step_idx
                    denoise_per_step.append(step_timing)
                step_idx += 1
        if inactive_mask is not None:
            x_t = x_t * inactive_mask  # t=0 endpoint: inactive dims land at 0 in both schemes
        t5 = sync_ts(_t)

        if _t:
            step_totals = [s.get("step_total_ms", 0.0) for s in denoise_per_step] if denoise_per_step else []
            denoising_loop_total_ms = (t5 - t_loop) * 1000
            mean_step_ms = (
                denoising_loop_total_ms / num_steps
                if use_static_cache and num_steps > 0
                else sum(step_totals) / len(step_totals)
                if step_totals
                else 0.0
            )
            timing["noise_init_ms"] = (t1 - t0) * 1000
            timing["build_prefix_ms"] = (t2 - t1) * 1000
            timing["build_prefix"] = build_prefix_timing
            timing["prefix_forward_ms"] = (t4 - t3) * 1000
            timing["denoising_loop_total_ms"] = denoising_loop_total_ms
            timing["denoising_loop"] = {
                "num_steps": num_steps,
                "backend": "compile" if use_static_cache else "dynamic_eager",
                "mean_step_ms": mean_step_ms,
                "per_step": [] if use_static_cache else denoise_per_step,
            }

        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep, timing: Optional[dict] = None):
        _t = timing is not None

        # Step A: embed state + noisy actions + timestep into expert input tokens
        t0 = sync_ts(_t)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)
        t1 = sync_ts(_t)

        # Step B: build 2D attention mask + position ids, reusing embed_suffix's suffix
        # masks (no recompute). Same builder as the static path.
        full_att_2d_masks, position_ids = self._denoise_2d_masks(prefix_pad_masks, suffix_pad_masks, suffix_att_masks)
        t2 = sync_ts(_t)

        # Step C: expert-only forward, reusing cached VLM KV (fill_kv_cache=False)
        t3 = sync_ts(_t)
        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        t4 = sync_ts(_t)

        # Step D: slice action tokens and project to velocity
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        t5 = sync_ts(_t)

        if _t:
            timing["embed_suffix_ms"] = (t1 - t0) * 1000
            timing["mask_build_ms"] = (t2 - t1) * 1000
            timing["expert_forward_ms"] = (t4 - t3) * 1000
            timing["out_proj_ms"] = (t5 - t4) * 1000
            timing["step_total_ms"] = (t5 - t0) * 1000

        return v_t

    def _build_denoise_masks(self, prefix_pad_masks: torch.Tensor, suffix_len: int):
        """Static-path entry: build the (step-invariant) suffix masks, then the 2D
        attention mask + position ids. Called once before the compiled denoise loop.

        The eager path doesn't go through here — it already has the suffix masks from
        ``embed_suffix`` and calls ``_denoise_2d_masks`` directly (avoiding a recompute).
        """
        suffix_pad_masks, suffix_att_masks = self._suffix_masks(
            prefix_pad_masks.shape[0], suffix_len, prefix_pad_masks.device
        )
        return self._denoise_2d_masks(prefix_pad_masks, suffix_pad_masks, suffix_att_masks)

    def _denoise_2d_masks(self, prefix_pad_masks, suffix_pad_masks, suffix_att_masks):
        """Build the suffix's 2D attention mask (suffix attends to [prefix KV | suffix])
        and position ids from already-built suffix masks.

        Single source of truth for this construction, shared by the eager path (reuses
        embed_suffix's masks) and ``_build_denoise_masks`` (static/compiled path).
        """
        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs for suffix (continue from last valid prefix position)
        action_rope_offset = self.config.action_rope_offset
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1 + action_rope_offset

        return full_att_2d_masks, position_ids

    def _predict_velocity_core(self, state, x_t, timestep, past_key_values, full_att_2d_masks, position_ids):
        """Denoising step with no timing, no getattr — suitable for torch.compile."""
        time_embs, suffix_embs, _, _ = self.embed_suffix(state, x_t, timestep)

        ada_cond = time_embs if self._use_adanorm else None
        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            ada_cond=ada_cond,
        )

        suffix_out = outputs_embeds[1][:, -self._n_action_steps :]
        return self.action_out_proj(suffix_out)


# ===========================================================================
# Tau0VLA Model (top-level)
# ===========================================================================


class Tau0VLAModel(PreTrainedModel):
    """Tau VLA model: Qwen3.5 VL backbone + Qwen3.5 action expert with flow matching."""

    config_class = Tau0VLAConfig
    base_model_prefix = "tau_vla"
    _no_split_modules = ["Qwen3_5DecoderLayerForExpert", "Qwen3_5DecoderLayerForVLM"]
    _tied_weights_keys = []
    _arch_keys = ("max_action_dim", "max_state_dim", "n_action_steps", "adanorm_time")
    supports_gradient_checkpointing = True

    def __init__(self, config: Tau0VLAConfig):
        super().__init__(config)
        self.config = config
        self.flow_matching = FlowMatching(config)

        # if not getattr(self.config, "use_lm_head", False):
        #     del self.flow_matching.qwenvl_with_expert.qwenvl.lm_head
        del self.flow_matching.qwenvl_with_expert.qwen_expert.lm_head

        torch.set_float32_matmul_precision("high")
        self.post_init()

    def get_input_embeddings(self):
        return self.flow_matching.qwenvl_with_expert.qwenvl.get_input_embeddings()

    def get_output_embeddings(self):
        return self.flow_matching.qwenvl_with_expert.qwenvl.get_output_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.flow_matching.qwenvl_with_expert.qwenvl.resize_token_embeddings(*args, **kwargs)

    def tie_weights(self, **kwargs):
        """Tie lm_head to embed_tokens after weight loading.

        Skip super() to avoid HF's deep submodule scan hitting deleted lm_head modules.
        Tau0VLAModel has no tied weights of its own (_tied_weights_keys = []);
        the only tie needed is qwenvl's lm_head <-> embed_tokens when use_lm_head=True.
        """
        if getattr(self.config, "use_lm_head", False):
            self.flow_matching.qwenvl_with_expert.qwenvl.tie_weights(**kwargs)

    @classmethod
    def validate_arch_compat(cls, saved_cfg, overrides: dict):
        """Raise if architecture-critical params in *overrides* conflict with *saved_cfg*."""
        for key in cls._arch_keys:
            if key not in overrides:
                continue
            saved_val = getattr(saved_cfg, key, None)
            new_val = overrides[key]
            if saved_val is not None and new_val is not None and saved_val != new_val:
                raise ValueError(
                    f"Architecture mismatch: checkpoint has {key}={saved_val}, "
                    f"but current args specify {key}={new_val}. "
                    f"Cannot change weight-shape params when loading a VLA checkpoint for SFT."
                )

    @classmethod
    def from_vlm_pretrained(cls, vlm_base_path, config):
        """Build Tau0VLA from a VLM base model, loading VLM pretrained weights."""
        model = cls(config)
        # Load VLM weights into the qwenvl sub-module
        sf_files = sorted(glob.glob(os.path.join(vlm_base_path, "model*.safetensors")))
        if sf_files:
            pretrained_sd = {}
            for f in sf_files:
                pretrained_sd.update(safetensors_load(f, device="cpu"))
            qwenvl = model.flow_matching.qwenvl_with_expert.qwenvl
            missing, unexpected = qwenvl.load_state_dict(pretrained_sd, strict=False, assign=True)
            if missing:
                logger.warning(f"VLM missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                logger.warning(f"VLM unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            logger.info(f"Loaded VLM pretrained weights from {vlm_base_path} ({len(sf_files)} shard(s))")
        else:
            logger.warning(f"No safetensors found in {vlm_base_path}, VLM weights are RANDOM!")
        model.tie_weights()
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override to call tie_weights after HF weight loading."""
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.tie_weights()
        return model

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        action=None,
        state=None,
        **kwargs,
    ) -> Tau0VLAOutput:
        # --- Action path (flow matching) ---
        if action is None or state is None:
            raise ValueError(
                "Tau0VLAModel.forward requires both `action` and `state`: the only training "
                "route is the flow-matching action route. For inference use `sample_action`."
            )

        actions = action
        if state.ndim == 3:
            state = state.squeeze(1)

        fm_losses = self.flow_matching.forward(
            state=state,
            actions=actions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            loss_type=self.config.loss_type,
            # Scheme C only: zero the flow input on inactive dims. Scheme A
            # (default) passes None and keeps the t*noise training input.
            action_mask=(
                kwargs.get("action_mask")
                if getattr(self.config, "vla_inactive_input_zero", False)
                else None
            ),
        )

        fm_losses = fm_losses[:, :, : self.config.action_dim]

        action_mask = kwargs.get("action_mask")
        if getattr(self.config, "use_action_mask_loss", True) and action_mask is not None:
            mask = action_mask[:, : self.config.action_dim].unsqueeze(1)
            fm_losses = fm_losses * mask
            active_dims = mask.sum(dim=-1) * fm_losses.shape[1]
            per_sample_loss = fm_losses.sum(dim=(1, 2)) / (active_dims.squeeze(-1) + 1e-8)

            _SPIKE_THRESH = 100.0
            spike_mask = per_sample_loss > _SPIKE_THRESH
            if spike_mask.any():
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_rank() == 0:
                    for idx in spike_mask.nonzero(as_tuple=True)[0][:3]:
                        logger.warning(
                            f"[spike] sample {idx.item()}: loss={per_sample_loss[idx].item():.2f}, "
                            f"active_dims={int(mask[idx].sum())}"
                        )
            per_sample_loss = torch.clamp(per_sample_loss, max=_SPIKE_THRESH)
            vla_loss = per_sample_loss.mean()
        else:
            vla_loss = fm_losses.mean()

        return Tau0VLAOutput(
            loss=vla_loss,
            vla_loss=vla_loss,
        )

    @torch.no_grad()
    def sample_action(self, batch: dict, timing: Optional[dict] = None) -> Tensor:
        """Inference interface compatible with openloop evaluation.

        Args:
            batch: dict with keys from ``build_vla_inference_data_dict`` plus ``state``.
                Required: input_ids, attention_mask, state.
                Optional: pixel_values, image_grid_thw.
            timing: optional dict; when provided, per-stage latency is recorded into it.

        Returns:
            actions: (batch_size, n_action_steps, action_dim)
        """
        state = batch["state"]
        if state.ndim == 3:
            state = state.squeeze(1)

        actions = self.flow_matching.sample_actions(
            state=state,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            # per-sample 40D mask (from data_spec / finch batch). None -> legacy
            # free integration; present -> inactive dims pinned to the training
            # input distribution each step (see sample_actions).
            action_mask=batch.get("action_mask"),
            timing=timing,
        )
        return actions[:, :, : self.config.action_dim]


# ===========================================================================
# Register with HuggingFace Auto classes
# ===========================================================================

AutoConfig.register("tau_vla", Tau0VLAConfig)
AutoModel.register(Tau0VLAConfig, Tau0VLAModel)
