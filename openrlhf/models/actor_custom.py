from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from transformers.deepspeed import HfDeepSpeedConfig

from .packing_utils import patch_for_block_diag_attn
from .utils import log_probs_from_logits, log_probs_from_logits_with_modulation, reset_position_ids
from openrlhf.models.actor import Actor


class ActorCustom(nn.Module):
    """
    Modification of Actor model to also be able to output a modifier to the base model
    """

    def __init__(
        self,
        pretrain_or_model,
        initial_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.initial_model = initial_model

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
            if packing_samples:
                assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
                model_type = getattr(self.model.config, "model_type", None)
                patch_for_block_diag_attn(model_type)
        else:
            self.model = pretrain_or_model

        # Custom new linear (or later NN) head in order to output modifier to logits
        new_layer = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, )

        print("NEW LAYER WEIGHT 1")
        print(new_layer.weight)

        nn.init.xavier_normal_(new_layer.weight) # Lower variance initialization, also consistent with my previous work

        print("NEW LAYER WEIGHT 2")
        print(new_layer.weight)
        # TODO: if still doing nonsense, later try actual 0s and ensure it works, at least in the first iteration
        # ALSO TODO: ensure that the sampling matches the prob under the model.

        # new_layer.weight.data *= 0 # TODO DEBUG ONLY REMOVE LATER
        #
        # print("NEW LAYER WEIGHT 3")
        # print(new_layer.weight)

        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)

        self.model.lm_head = new_layer



    @torch.no_grad()
    # def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
    #     Tuple[torch.LongTensor, torch.LongTensor],
    #     Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    # ]:
    def generate(self, input_ids, *args, attention_mask=None,
                 condition_twist_on_tokens=None, **kwargs):
        r"""
        CUSTOM. Only sample available right now.
        """
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        # if kwargs.get("max_new_tokens", None):
        #     generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        # if kwargs.get("max_length", None):
        #     generate_args["max_length"] = kwargs.get("max_length")

        action_mask, attention_mask, sequences = self.custom_generate(
            attention_mask, generate_args, input_ids, **kwargs)

        # sequences, attention_mask, action_mask = self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

        # print("LM LOGITS SHAPE")
        # print(lm_logits.shape)
        # print("SEQ SHAPE")
        # print(sequences.shape)

        return sequences, attention_mask, action_mask

    def custom_generate(self, attention_mask, generate_args, input_ids, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens")
        max_len = max_new_tokens + input_ids.shape[-1]
        print(input_ids.shape)
        sequences = input_ids.clone()
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]
        while sequences.shape[-1] < max_len:

            lm_logits = self.forward(
                sequences,
                num_actions=max_new_tokens,
                # TODO: note this might cause some issues; keeping it simple for now
                attention_mask=attention_mask,
                # condition_twist_on_tokens=condition_twist_on_tokens
                use_for_generation=True
            )

            next_token_logits = lm_logits[:, -1, :]

            # sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # update generated ids, model inputs, and length for next step
            sequences = torch.cat([sequences, next_tokens[:, None]], dim=-1)
            sequences, attention_mask, action_mask = self.process_sequences(
                sequences, input_ids.size(1), eos_token_id, pad_token_id)
        return action_mask, attention_mask, sequences

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # print("--Sequences before modification--")
        # print(sequences)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #

        # eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        # sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # TODO The above may be the problem
        # Summary of what I think is the problem: essentially if you reach the max sequence length in generation
        # All the rest of the code does calculations based on this token which was manually inserted
        # The problem here is that the eos token may be very different in probability from what the model would actually generate
        # And this causes problems specifically with the KL divergence from the base policy, because the KL is now calculated on the EOS token
        # Which may have a very different value on the current policy vs ref/base policy
        # And of course, since the EOS token is not actually being sampled
        # This can also cause the KL div to go negative, and wildly so
        # TODO So I think we can simply just not have the EOS token at the end (possibly just remove the above 2 lines), and things should work ok. Not sure what problems this might cause though, by removing the final EOS processing
        # Also not sure what other problems may arise
        # TODO open an issue on the OpenRLHF repo
        # TODO After resolving this issue, keep stepping through code and checking elsewhere what is going on

        # print("--Sequences after modification--")
        # print(sequences)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        # first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask



    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        use_for_generation=False
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
        else:
            # reset the positions for packed samples
            position_ids = reset_position_ids(attention_mask)
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            base_output = self.initial_model.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        # log_psi
        modulation = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        if use_for_generation: # In the generation loop, need all logits for all vocab. Otherwise just need evaluation of the particular log_p
            return_all_vocab = True
            log_probs = log_probs_from_logits_with_modulation(
                base_output["logits"], modulation["logits"], sequences, # UNUSED HERE, TODO refactor to make this cleaner
                return_all_vocab=return_all_vocab
            )
            return log_probs # actually only need the very last one of this... [:, -1]

        else:
            return_all_vocab = False

            log_probs = log_probs_from_logits_with_modulation(base_output["logits"][:, :-1, :], modulation["logits"][:, :-1, :], sequences[:, 1:], return_all_vocab=return_all_vocab)

            # output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

            assert not return_output
            # if return_output:
            #     return output if num_actions is None else (log_probs[:, -num_actions:], output)
            # else:

            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        raise NotImplementedError
        self.model.print_trainable_parameters()


class ActorCritic(nn.Module):
    """
    Actor with shared parameters with critic
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                raise NotImplementedError # TODO: Make sure that the value head works properly here too
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
            if packing_samples:
                assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
                model_type = getattr(self.model.config, "model_type", None)
                patch_for_block_diag_attn(model_type)
        else:
            self.model = pretrain_or_model

        # Custom new linear (or later NN) head in order to output VALUE/CRITIC
        # new_layer = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, )
        new_layer = nn.Linear(self.model.lm_head.in_features, 1)

        # nn.init.xavier_normal_(new_layer.weight) # Lower variance initialization, also consistent with my previous work
        # NOTE: with out features 1 this xavier init does not actually seem to be low variance at all for a value head...

        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)

        self.critic_head = new_layer



    @torch.no_grad()
    # def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
    #     Tuple[torch.LongTensor, torch.LongTensor],
    #     Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    # ]:
    def generate(self, input_ids, *args, attention_mask=None,
                 condition_twist_on_tokens=None, **kwargs):
        r"""
        CUSTOM. Only sample available right now.
        """
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # max_new_tokens = kwargs.get("max_new_tokens")
        # max_len = max_new_tokens + input_ids.shape[-1]
        # print(input_ids.shape)

        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        # Call generate
        sequences = self.model.generate(**generate_args)

        sequences, attention_mask, action_mask = self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

        # print("LM LOGITS SHAPE")
        # print(lm_logits.shape)
        # print("SEQ SHAPE")
        # print(sequences.shape)

        return sequences, attention_mask, action_mask

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # print("--Sequences before modification--")
        # print(sequences)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #

        # eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        # sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # TODO The above may be the problem
        # Summary of what I think is the problem: essentially if you reach the max sequence length in generation
        # All the rest of the code does calculations based on this token which was manually inserted
        # The problem here is that the eos token may be very different in probability from what the model would actually generate
        # And this causes problems specifically with the KL divergence from the base policy, because the KL is now calculated on the EOS token
        # Which may have a very different value on the current policy vs ref/base policy
        # And of course, since the EOS token is not actually being sampled
        # This can also cause the KL div to go negative, and wildly so
        # TODO So I think we can simply just not have the EOS token at the end (possibly just remove the above 2 lines), and things should work ok. Not sure what problems this might cause though, by removing the final EOS processing
        # Also not sure what other problems may arise
        # TODO open an issue on the OpenRLHF repo
        # TODO After resolving this issue, keep stepping through code and checking elsewhere what is going on

        # print("--Sequences after modification--")
        # print(sequences)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        # first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask



    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
        else:
            # reset the positions for packed samples
            position_ids = reset_position_ids(attention_mask)
        position_ids.masked_fill_(attention_mask == 0, 1)

        # print("INSPECTION OF FORWARD CALL")
        # print(sequences)
        # print(sequences.shape)
        # print(position_ids.shape)
        # print(attention_mask.shape) # FIgure out where this attention mask is coming from and why it doesn't work with position_ids...

        output = self.model(sequences, attention_mask=attention_mask,
                            position_ids=position_ids, output_hidden_states=True)


        log_probs = log_probs_from_logits(output["logits"][:, :-1, :],
                                          sequences[:, 1:])

        last_hidden_state = output.hidden_states[-1]

        if last_hidden_state.device != self.critic_head.weight.device:
            last_hidden_state = last_hidden_state.to(
                self.critic_head.weight.device)

        value = self.critic_head(last_hidden_state).squeeze(-1)

        # print(value.shape)
        # print(log_probs.shape)
        # print(log_probs[:, -num_actions:].shape)
        # # TODO check the return statement below then modify this wherever else in the code needs to be modified also.
        # 1/0

        if return_output:
            return output if num_actions is None else (
            log_probs[:, -num_actions:], output, value[:, -num_actions-1:-1])
            # Right, reason why we do this: in the last token state, where you have the full generation: you are no longer taking any actions. You don't need the value in that state, to compare against rewards or whatever, because there are no further actions to be taken. Once all the 8 or whatever initial prompt tokens are given, your model outputs a value. This is the value for the initial state, which you then take an action corresponding to a token generation, then for that generated token, you do r + next val - curr val, and the curr value is the one based on the 8 prompt tokens you generated, ie in the 8th position (index 7 via 0 indexing). So yeah, the 7 index (8th value counting from 1) should already be used.
        else:
            return log_probs[:, -num_actions:], value[:, -num_actions-1:-1]


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        raise NotImplementedError
        self.model.print_trainable_parameters()
