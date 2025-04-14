from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
try:
    from transformers.deepspeed import HfDeepSpeedConfig
except:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# from .packing_utils import patch_for_block_diag_attn
from .utils import reset_position_ids
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
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

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # packing samples using Flash Attention 2
    if packing_samples:
        raise NotImplementedError # check the latest OpenRLHF repo
        # assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
        # model_type = getattr(model.config, "model_type", None)
        # patch_for_block_diag_attn(model_type)

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            # print("VALUES/CRITIC CHECK")
            # print(values)

            if self.packing_samples:
                reward = values
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_reward_model_custom(
    base_pretrained_class, rm_name, tokenizer_base, config, separatequeryanswer=False, max_new_tokens=None,
    strip_question_chat_template_fn=None
):
    class RewardModel(base_pretrained_class):
        supports_gradient_checkpointing = True

        def __init__(self):
            super().__init__(config)

            from transformers import AutoModelForSequenceClassification, \
                AutoTokenizer

            self.rm = AutoModelForSequenceClassification.from_pretrained(
                rm_name)
            self.tokenizer_RM = AutoTokenizer.from_pretrained(rm_name)
            self.tokenizer_base = tokenizer_base  # TODO ensure this works
            self.max_new_tokens = max_new_tokens
            if max_new_tokens is not None:
                assert self.max_new_tokens > 0


        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask=None,
            return_output=False,
        ) -> torch.Tensor:

            if separatequeryanswer:
                assert max_new_tokens is not None

                print("Toy RLHF inspection")
                print(input_ids.shape)
                print(max_new_tokens)

                # print("input_ids")
                # print(input_ids)

                # question_seq = input_ids[:, lstrip_from_question_n_tokens : -self.max_new_tokens - rstrip_from_question_n_tokens]
                # answer_seq = input_ids[:, -self.max_new_tokens + strip_from_answer_n_tokens : ]
                # question_seq = input_ids[:, : -self.max_new_tokens]
                # answer_seq = input_ids[:, -self.max_new_tokens : ]



                # print("seqs")
                # print(question_seq)
                # print(answer_seq)

                # text_question = self.tokenizer.batch_decode(question_seq,
                #                                        skip_special_tokens=True)
                # text_answer = self.tokenizer.batch_decode(answer_seq,
                #                                      skip_special_tokens=True)

                text = self.tokenizer_base.batch_decode(input_ids, skip_special_tokens=True)

                # print(text)

                qa_list = list(map(strip_question_chat_template_fn, text))

                # print(qa_list)
                #
                # print(list(zip(*qa_list)))

                text_question, text_answer = map(list, zip(*qa_list))
                # text_questions = list(map(lambda x: x[0], qa_list))
                # text_answers = list(map(lambda x: x[1], qa_list))

                # print("attention_mask")
                # print(attention_mask)

                # text_question = list(map(lambda x: x.removeprefix('user\n').removesuffix('\nassistant\n'), texts))
                print("text questions and answers")
                print(text_question)
                print(text_answer)

                device = self.rm.device

                if rm_name == "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft":
                    messages = []
                    for q, a in zip(text_question, text_answer):
                        message = [
                            {'role': 'user', 'content': q},
                            {'role': 'assistant', 'content': a}
                        ]
                        messages.append(self.tokenizer_RM.apply_chat_template(message, tokenize=False))

                    kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
                    tokens = self.tokenizer_RM(messages, **kwargs)

                    input_ids = tokens["input_ids"].to(device)
                    attention_mask = tokens["attention_mask"].to(device)

                    with torch.no_grad():
                        rewards_tensor = self.rm(input_ids=input_ids, attention_mask=attention_mask)[0]
                        r = rewards_tensor.detach().squeeze(-1)  # squeeze last dim if needed

                else:
                    inputs = self.tokenizer_RM(text_question, text_answer,
                                          return_tensors="pt",
                                          padding=True
                                          )

                    # print("inputs to rm")
                    # print(inputs)

                    # output_len = answer_seq.shape[-1]
                    # print(output_len)

                    # print("device")
                    # print(device)
                    # print(question_seq.device)

                    inputs = {key: value[:, :self.max_new_tokens * 2].to(device) for
                              key, value in
                              inputs.items()}  # Truncate to no more than 2x output len, otherwise can have some crazy tokenizations.

                    # print("inputs to rm 2")
                    # print(inputs)

                    with torch.no_grad():
                        r = self.rm(**inputs).logits.squeeze(-1).detach()

                print("reward")
                print(r)
                # 1/0

            else:

                assert return_output == False
                # print("--FORWARD CALL--")
                # print(input_ids.device)
                text = self.tokenizer_base.batch_decode(input_ids)
                # print(text)
                tokens = self.tokenizer_RM(text, return_tensors="pt", padding=True)
                # print(tokens)
                # print(tokens['input_ids'].device)
                # print(tokens['attention_mask'].device)
                r = self.rm(input_ids=tokens['input_ids'].to(input_ids.device),
                      attention_mask=tokens['attention_mask'].to(input_ids.device)).logits.squeeze()
                # print(rew)
                # print("--MEAN OF REWARDS--")
                # print(rew.mean())
                # print("--END FORWARD CALL--")

            return r

    return RewardModel()


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1] # Ok here it is, this part is ok then
            num_actions = action_mask.size(1)

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if return_output:
                return outputs if num_actions is None else (values[:, -num_actions:], outputs)
            else:
                return values[:, -num_actions:]

    return CriticModel
