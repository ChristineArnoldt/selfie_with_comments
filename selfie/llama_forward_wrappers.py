# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def model_forward_interpret( # High-level wrapper for the forward pass
    model=None,  # The model object used for forward pass
    input_ids: torch.LongTensor = None,  # Input token IDs to process
    attention_mask: Optional[torch.Tensor] = None,  # Mask to ignore padding tokens during attention
    position_ids: Optional[torch.LongTensor] = None,  # Optional positional encoding IDs
    past_key_values: Optional[List[torch.FloatTensor]] = None,  # Cached states for faster decoding
    inputs_embeds: Optional[torch.FloatTensor] = None,  # Pre-computed input embeddings (alternative to input_ids)
    labels: Optional[torch.LongTensor] = None,  # Token labels for computing the loss
    use_cache: Optional[bool] = None,  # Whether to cache past key values for efficient decoding
    output_attentions: Optional[bool] = None,  # Whether to return attention weights
    output_hidden_states: Optional[bool] = None,  # Whether to return hidden states for all layers
    return_dict: Optional[bool] = None,  # Whether to return outputs in a structured dictionary
    insert_info=None,  # Optional interpretability info to modify hidden states
    output_pre_mlp_states=False,  # Whether to include pre-MLP states in the output
) -> Union[Tuple, CausalLMOutputWithPast]:  # Return type can be a tuple or a structured output
    """
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Tokens with indices set to `-100` are ignored.

    Returns:
        Outputs or loss, depending on inputs and return configuration.
    """

    # Use the specified `output_attentions`, `output_hidden_states`, and `return_dict` or fallback to model config
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # Call the `model_model_forward_interpret` function to perform the main forward pass
    # This function handles most of the logic, including interpretability modifications
    outputs, all_original_hidden_states = model_model_forward_interpret(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        insert_info=insert_info,  # Pass interpretability modifications if any
    )

    hidden_states = outputs[0]  # Extract the last hidden states from the output

    if model.config.pretraining_tp > 1:
        # Handle model parallelism if pretraining involves multiple tensor parallel groups
        # Split the model's vocabulary across groups for logits computation
        lm_head_slices = model.lm_head.weight.split(model.vocab_size // model.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(model.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)  # Concatenate the logits across groups
    else:
        # Standard logits computation using the model's `lm_head`
        logits = model.lm_head(hidden_states)

    logits = logits.float()  # Convert logits to `float` for numerical stability

    loss = None  # Initialize loss as `None`

    if labels is not None:
        # If labels are provided, compute the language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()  # Shift logits for left-to-right prediction
        shift_labels = labels[..., 1:].contiguous()  # Shift labels to align with logits
        loss_fct = CrossEntropyLoss()  # Use CrossEntropyLoss for language modeling
        shift_logits = shift_logits.view(-1, model.config.vocab_size)  # Flatten logits
        shift_labels = shift_labels.view(-1)  # Flatten labels
        shift_labels = shift_labels.to(shift_logits.device)  # Ensure labels are on the same device as logits
        loss = loss_fct(shift_logits, shift_labels)  # Compute loss

    if not return_dict:
        # If `return_dict` is False, return a tuple of outputs
        output = (logits,) + outputs[1:]  # Combine logits with other outputs
        return (loss,) + output if loss is not None else output  # Include loss if computed

    # Handle optional inclusion of pre-MLP states in the output
    if output_pre_mlp_states:
        if output_attentions:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), all_original_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), all_original_hidden_states

    if output_attentions:
        # If `output_attentions` is True, include attentions in the output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        # Otherwise, return the outputs without attentions
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
# fine-grained control over the internal workings of the model
# especially for interpretability-focused modifications
@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def model_model_forward_interpret( # lower level implementation of the forward pass
    model = None,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    insert_info = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # Set default values for output_attentions, output_hidden_states, and use_cache based on the model's config
    output_attentions = output_attentions if output_attentions is not None else model.model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.model.config.use_cache
    return_dict = return_dict if return_dict is not None else model.model.config.use_return_dict

    # Ensure that either `input_ids` or `inputs_embeds` are provided, but not both
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape  # Get batch size and sequence length from input_ids
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape  # Get batch size and seq length from inputs_embeds
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # Initialize sequence length with past context and past key values length as 0
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        # If past_key_values are provided, calculate their sequence length
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        # Generate position IDs if not provided, accounting for past_key_values
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        ).unsqueeze(0).view(-1, seq_length)  # Shape: (batch_size, sequence_length)
    else:
        # Ensure position_ids have the correct shape and data type
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        # Generate input embeddings if inputs_embeds are not provided
        inputs_embeds = model.model.embed_tokens(input_ids)

    # If no attention mask is provided, default to a mask with all 1s (no padding)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        # If the attention mask contains zeros, treat them as padding positions
        padding_mask = attention_mask if 0 in attention_mask else None

    # Prepare the decoder's attention mask, accounting for past_key_values
    attention_mask = model.model._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    # Initialize hidden states and original hidden states for interpretability tracking
    hidden_states = inputs_embeds
    original_hidden_states = (inputs_embeds, inputs_embeds)

    if model.model.gradient_checkpointing and model.model.training:
        # Disable caching when gradient checkpointing is enabled to avoid conflicts
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # Initialize containers for optional outputs
    all_hidden_states = () if output_hidden_states else None
    all_original_hidden_states = ()
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Iterate through the decoder layers
    for idx, decoder_layer in enumerate(model.model.layers):
        if hidden_states.shape[1] > 1 and insert_info is not None:
            # Apply interpretability modifications if `insert_info` is provided
            for batch_item_idx in range(len(insert_info)):
                if idx in insert_info[batch_item_idx].keys():
                    if insert_info[batch_item_idx]['replacing_mode'] == 'addition':
                        # Add overlay information to hidden states
                        hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :] += \
                            insert_info[batch_item_idx]['overlay_strength'] * insert_info[batch_item_idx][idx][1].to(hidden_states.device)
                    elif insert_info[batch_item_idx]['replacing_mode'] == 'normalized':
                        # Combine overlay information with hidden states in a weighted manner
                        hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :] = \
                            insert_info[batch_item_idx]['overlay_strength'] * insert_info[batch_item_idx][idx][1].to(hidden_states.device) + \
                            (1 - insert_info[batch_item_idx]['overlay_strength']) * \
                            hidden_states[batch_item_idx:batch_item_idx+1, insert_info[batch_item_idx][idx][0], :]

        # Save hidden states if requested
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_original_hidden_states += (original_hidden_states,)

        # Retrieve cached past key values for the current layer if available
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if model.model.gradient_checkpointing and model.model.training:
            # Use gradient checkpointing for memory efficiency during training
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)
                return custom_forward
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
            )
        else:
            # Perform a standard forward pass through the decoder layer
            layer_outputs = decoder_layer_forward_interpret(
                hidden_states,
                decoder_layer=decoder_layer,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        layer_outputs, original_hidden_states = layer_outputs
        hidden_states = layer_outputs[0]

        if use_cache:
            # Cache the outputs for autoregressive decoding
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            # Save attention weights if requested.
            all_self_attns += (layer_outputs[1],)

    # Apply the model's final normalization layer to the last hidden state
    hidden_states = model.model.norm(hidden_states)

    # Save the last hidden state if requested
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
        all_original_hidden_states += (original_hidden_states,)

    # Prepare the outputs for return
    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    ), all_original_hidden_states



# Function to process a single decoder layer during forward pass, interpreting its behavior
def decoder_layer_forward_interpret(
    hidden_states: torch.Tensor,  # Input tensor of shape (batch, seq_len, embed_dim), representing token embeddings
    decoder_layer=None,  # The decoder layer object that implements specific transformations
    attention_mask: Optional[torch.Tensor] = None,  # Mask for attention mechanism; used to avoid attending to padding tokens
    position_ids: Optional[torch.LongTensor] = None,  # Positional encoding indices for sequence elements
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # Cached keys and values for attention (used in decoding for efficiency)
    output_attentions: Optional[bool] = False,  # Whether to return attention weights for interpretability or debugging
    use_cache: Optional[bool] = False,  # Whether to return the updated past_key_values for reuse in future computations
    padding_mask: Optional[torch.LongTensor] = None,  # Mask to specify positions that should be ignored (e.g., padding)
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
  
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    # Returns: Tuple containing updated hidden states, and optionally attention weights and past_key_values
   
    # Store the input hidden states for use in the residual connection
    residual = hidden_states

    # Apply layer normalization to stabilize and scale the hidden states
    hidden_states = decoder_layer.input_layernorm(hidden_states)

    # --- Self Attention Step ---
    # Pass normalized hidden states through the self-attention mechanism
    # print(type(hidden_states), hidden_states.shape)
    hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
        hidden_states=hidden_states,  # The normalized hidden states as input
        attention_mask=attention_mask,  # The attention mask to handle padding or causal masking
        position_ids=position_ids,  # Positional encodings for the sequence
        past_key_value=past_key_value,  # Cached key and value projections for reuse
        output_attentions=output_attentions,  # Whether to output attention weights
        use_cache=use_cache,  # Whether to return the updated past_key_values
        padding_mask=padding_mask,  # Padding mask for additional masking requirements
    )

    # Ensure the updated hidden states remain on the same device as the residual tensor
    hidden_states = hidden_states.to(residual.device)
    # print(hidden_states.device)
    # print(residual.device)

    # Add the residual (skip connection) to the output of the self-attention layer
    # This helps retain the original information and supports gradient flow
    hidden_states = residual + hidden_states

    # --- Fully Connected Step ---
    # Update the residual to store the current state
    residual = hidden_states
    # Apply another layer normalization after attention for stability
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    # Save the hidden states before feeding them to the MLP (useful for interpretability)
    pre_mlp = hidden_states
    # Pass the normalized hidden states through the multi-layer perceptron (MLP)
    hidden_states = decoder_layer.mlp(hidden_states)

    # Ensure the output of the MLP remains on the same device as the residual tensor
    hidden_states = hidden_states.to(residual.device)
    # print(hidden_states.device)
    # print(residual.device)
    
    # Save the original states (pre-MLP normalized states and residual) for interpretability
    original_hidden_states = (pre_mlp, residual)
    # Add the residual connection to the MLP output
    hidden_states = residual + hidden_states

    # --- Prepare Outputs ---
    # Start building the output tuple with the updated hidden states
    outputs = (hidden_states,)

    # If requested, append the attention weights to the output for interpretability
    if output_attentions:
        outputs += (self_attn_weights,)

    # If requested, append the cached keys and values for decoding speedup
    if use_cache:
        outputs += (present_key_value,)

    # Return the outputs (updated hidden states and optionally other details) and the original hidden states for interpretation
    return outputs, original_hidden_states