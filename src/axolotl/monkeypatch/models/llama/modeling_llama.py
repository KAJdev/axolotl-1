import contextlib
import inspect
import types

from torchtune.training import OffloadActivations
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.monkeypatch.unsloth_ import detab_code

HF_MODEL_OUTPUTS = """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
""".lstrip()

PATCHED_HF_MODEL_OUTPUTS = """
        with self.act_offloading_ctx_manager:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
""".lstrip()

LCE_MODEL_OUTPUTS = """
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
""".lstrip()

PATCHED_LCE_OUTPUTS = """
    with self.act_offloading_ctx_manager:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
""".lstrip()

HF_GA_FORWARD_1 = """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
""".lstrip()

PATCHED_HF_GA_FORWARD_1 = """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # remove num_items_in_batch otherwise self.model attempts to pass it to flash_attention
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
""".lstrip()

HF_GA_FORWARD_2 = """
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
""".lstrip()

PATCHED_HF_GA_FORWARD_2 = """
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, num_items_in_batch=num_items_in_batch, **kwargs)
""".lstrip()


class AxolotlLlamaForCausalLM(LlamaForCausalLM):
    act_offloading_ctx_manager = contextlib.nullcontext()

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        forward_source = inspect.getsource(LlamaForCausalLM.forward)
        self.forward = types.MethodType(
            compile(forward_source, "<forward>", "exec"), self
        )

    @classmethod
    def enable_act_offloading(cls):
        forward_source = inspect.getsource(cls.forward)
        forward_source = forward_source.replace(
            HF_MODEL_OUTPUTS, PATCHED_HF_MODEL_OUTPUTS
        )
        # replace forward method with patched version
        cls.forward = types.MethodType(
            compile(forward_source, "<llama_forward_w_act_offloading>", "exec"), cls
        )
        cls.act_offloading_ctx_manager = OffloadActivations()

    @classmethod
    def enable_liger_fce(cls, enable_act_offloading=True):
        from liger_kernel.transformers.model.llama import (
            lce_forward as llama_lce_forward,
        )

        if enable_act_offloading:
            lce_source = inspect.getsource(llama_lce_forward)
            lce_source = lce_source.replace(LCE_MODEL_OUTPUTS, PATCHED_LCE_OUTPUTS)
            # replace forward method with patched version
            cls.forward = types.MethodType(
                compile(lce_source, "<llama_lce_forward_w_act_offloading>", "exec"),
                cls,
            )
        else:
            cls.forward = types.methodType(llama_lce_forward, cls)

    @classmethod
    def patch_hf_ga(cls):
        # bugfix patch for gradient accumulation
        forward_source = inspect.getsource(cls.forward)
        forward_source = forward_source.replace(
            HF_GA_FORWARD_1, PATCHED_HF_GA_FORWARD_1
        )
        forward_source = forward_source.replace(
            HF_GA_FORWARD_2, PATCHED_HF_GA_FORWARD_2
        )
        forward_source = detab_code(forward_source)
        # replace forward method with patched version
        cls.forward = types.MethodType(
            compile(forward_source, "<llama_forward_ga_fix>", "exec"), cls
        )


def replace_auto_model():
    from transformers import LlamaConfig
    from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING

    MODEL_FOR_CAUSAL_LM_MAPPING[LlamaConfig] = AxolotlLlamaForCausalLM

    return AxolotlLlamaForCausalLM
