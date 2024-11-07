#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class DisasterAttentionModel2(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(DisasterAttentionModel2, self).__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.conv_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        # Input shape: (B, C, L) where L=4096
        B, C, L = x1.shape

        # Transpose for multihead attention (B, L, C)
        x1_flat = x1.permute(0, 2, 1)  # (B, L, C)
        x2_flat = x2.permute(0, 2, 1)  # (B, L, C)

        attended_output, _ = self.cross_attention(query=x1_flat, key=x2_flat, value=x2_flat)

        # Restore original shape for further processing (B, C, L)
        attended_output = attended_output.permute(0, 2, 1)

        # Compute the difference and apply 1D convolutional projection
        combined_features = x2 - attended_output
        combined_features = self.conv_proj(combined_features)

        return combined_features

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.cross_attn2 = DisasterAttentionModel2(dim=256, num_heads=8)

    def get_cross_atten2(self):
        return self.cross_attn2

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
