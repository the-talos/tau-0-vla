from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel,
    Qwen3VLMoeVisionModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from tau0_vla.utils.device import resolve_compute_dtype
from tau0_vla.utils.utils import rank0_print

logger = logging.get_logger(__name__)


# Supports batch_size > 1, and both MQA and GQA.
def flash_attention_forward_multi_batch(
    module: torch.nn.Module,
    query: torch.Tensor,  # [batch, nheads, seq_len, head_dim]
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # [batch, seq_len] bool or int (1=valid, 0=pad)
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # Check output_attentions (FA2 does not support it).
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            f"Tensor query has a zero dimension: {query.shape}.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    # === Step 0: handle dtype (LayerNorm runs in fp32) ===
    if query.dtype == torch.float32:
        target_dtype = resolve_compute_dtype(module, query)
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    # === Step 1: reshape to what FA2 wants, [batch, seq_len, nheads, head_dim] ===
    # Note: under GQA/MQA q_nheads != kv_nheads, which FA2 supports natively.
    query = query.transpose(
        1, 2
    ).contiguous()  # [batch, q_nheads, seq_len, head_dim] -> [batch, seq_len, q_nheads, head_dim]
    key = key.transpose(
        1, 2
    ).contiguous()  # [batch, kv_nheads, seq_len, head_dim] -> [batch, seq_len, kv_nheads, head_dim]
    value = value.transpose(
        1, 2
    ).contiguous()  # [batch, kv_nheads, seq_len, head_dim] -> [batch, seq_len, kv_nheads, head_dim]

    batch_size, seq_len, q_nheads, head_dim = query.shape
    kv_nheads = key.shape[2]

    # === Step 2: build the padding mask ===
    if attention_mask is None:
        # No mask means every position is valid.
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=query.device)
    elif attention_mask.dtype != torch.bool:
        # Accept an int mask (1/0) as well.
        attention_mask = attention_mask.to(dtype=torch.bool, device=query.device)

    # === Step 3: compute cu_seqlens and max_seqlen (RIGHT-PADDING ONLY) ===
    # Each sample's true valid length.
    seq_lengths = attention_mask.sum(dim=1)  # [batch]

    # The longest valid sequence once padding is removed;
    # flash_attn_varlen_func needs it to compute attention correctly.
    max_seqlen = seq_lengths.max().item()

    cu_seqlens = torch.cat(
        [
            torch.tensor([0], device=query.device, dtype=torch.int32),
            seq_lengths.cumsum(dim=0).to(dtype=torch.int32),
        ]
    )  # [batch + 1]

    # === Step 4: compress away the padding (drop invalid tokens) ===
    # Flatten to [batch * seq_len, ...]
    query_flat = query.reshape(-1, q_nheads, head_dim)
    key_flat = key.reshape(-1, kv_nheads, head_dim)
    value_flat = value.reshape(-1, kv_nheads, head_dim)
    mask_flat = attention_mask.reshape(-1)  # [batch * seq_len]

    # Keep the valid tokens only.
    query_compressed = query_flat[mask_flat]  # [total_tokens, q_nheads, head_dim]
    key_compressed = key_flat[mask_flat]  # [total_tokens, kv_nheads, head_dim]
    value_compressed = value_flat[mask_flat]  # [total_tokens, kv_nheads, head_dim]
    total_tokens = query_compressed.shape[0]

    if total_tokens == 0:
        raise ValueError("All tokens are padding. Cannot compute attention.")

    # === Step 5: call FlashAttention-2 (GQA supported natively) ===
    window_size = (-1, -1)
    if sliding_window is not None:
        window_size = (sliding_window, sliding_window)  # FlashAttention-2's sliding window is symmetric

    # Note: flash_attn_varlen_func takes the scaling factor as softmax_scale.
    softmax_scale = scaling if scaling is not None else head_dim**-0.5

    attn_output_compressed = flash_attn_varlen_func(
        q=query_compressed,
        k=key_compressed,
        v=value_compressed,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        causal=True,
        window_size=window_size,
        softmax_scale=softmax_scale,
        # softcap=softcap,
        # alibi_slopes=None,
        # deterministic=False,
    )  # [total_tokens, q_nheads, head_dim]

    # === Step 6: restore the padding, back to the original shape ===
    attn_output_flat = torch.zeros(
        batch_size * seq_len,
        q_nheads,
        head_dim,
        dtype=attn_output_compressed.dtype,
        device=attn_output_compressed.device,
    )
    attn_output_flat[mask_flat] = attn_output_compressed
    attn_output = attn_output_flat.reshape(batch_size, seq_len, q_nheads, head_dim)

    # === Step 7: back to the standard shape [batch, q_nheads, seq_len, head_dim] ===
    # attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


# The original implementation: flattened data with batch_size=1.


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    # rank0_print("qwen2vl flash-attn forward")
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward_multi_batch(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],  # must be a [batch, seq_len] padding mask
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # rank0_print("qwen3vl flash-attn forward")
    input_shape = hidden_states.shape[:-1]  # [batch, seq_len]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # rope needs the reshape
    # [batch, seq_len, n_heads, head_dim] --> [batch, n_heads, seq_len, head_dim]
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward_multi_batch(
        self,
        query_states,  # [batch, n_heads, seq_len, head_dim]
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)  # [batch, seq_len, dim]
    return attn_output, attn_weights


def return_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, **kwargs):
    return attention_mask


def replace_qwen_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen2vl_forward
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = return_mask
    ## qwen2_5_vl
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2vl_forward
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = return_mask
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = return_mask
    ## qwen3vl
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3vl_forward
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask
    ## qwen3vl moe
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = qwen3vl_forward
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = return_mask
    rank0_print("Replace eagle attention to flash-attention.")


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    rank0_print("Vision Module - Attention Blocks:")
    rank0_print(f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}")
    rank0_print(f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}")
    rank0_print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(param.requires_grad for param in self.language_model.embed_tokens.parameters())
    rank0_print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    rank0_print(f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}")
    rank0_print(f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}")


Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters
