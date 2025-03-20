import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.models.actor_custom import ActorCustom, ActorCritic
from openrlhf.trainer import PPOTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.models.model import _get_reward_model_custom
from openrlhf.utils.utils import get_info_name_str


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )
    # Freeze initial model
    # This doesn't make a difference normally, but for my CustomActor
    # where I take this in as an argument, then the optimizer will optimize these
    # and this has two undesired effects 1) I'm basically adding parameters/capacity to the twist architecture
    # 2) More problematic is that I would then be modifying the initial model, so things like KL to prior
    # and F_q and G_q evaluations are all messed up.
    for param in initial_model.parameters():
        param.requires_grad = False
    get_tokenizer(args.pretrain, initial_model.model, "left", strategy)

    if args.shared_actorcritic:

        assert not args.actor_modulates_base # not yet implemented
        actor = ActorCritic(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        critic = None

    else:
        if args.actor_modulates_base:
            actor = ActorCustom(
                args.pretrain,
                initial_model=initial_model,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                additional_sd_divider=args.additional_sd_divider
            )
        else:
            # configure model
            # load huggingface model
            actor = Actor(
                args.pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
            )

        if args.no_critic:
            critic = None

        else:
            critic = get_llm_for_sequence_regression(
                args.critic_pretrain,
                "critic",
                normalize_reward=args.normalize_reward,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=False),
                value_head_prefix=args.value_head_prefix,
                init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
            )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    if not args.remote_rm_url:
        if args.reward_pretrain == "nicholasKluge/ToxicityModel":
            print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            reward_model = _get_reward_model_custom(base_pretrained_class, rm_name,
                                             tokenizer=tokenizer, config=config)

        elif args.reward_pretrain in ["OpenAssistant/reward-model-deberta-v3-base", "OpenAssistant/reward-model-deberta-v3-large-v2"]:
            print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            strip_question_chat_template_fn = None
            if args.apply_chat_template:
                if args.pretrain in ["HuggingFaceTB/SmolLM-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-3B-Instruct" ]:
                    def strip_question_chat_template_fn(text):
                        question, answer = text.split('assistant\n')
                        question = question.split('user\n')[-1].strip('\n')
                        # return text.removeprefix('user\n').removesuffix('\nassistant\n')
                        return question, answer
                else:
                    raise NotImplementedError
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer=tokenizer,
                config=config,
                separatequeryanswer=True,
                max_new_tokens=args.generate_max_len,
                strip_question_chat_template_fn=strip_question_chat_template_fn
            )



        else:
            if args.custom_single_prompt:
                raise NotImplementedError # Below does not necessarily work with my custom reward models


            reward_model = get_llm_for_sequence_regression(
                args.reward_pretrain,
                "reward",
                normalize_reward=args.normalize_reward,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=strategy.get_ds_train_config(is_actor=False),
                value_head_prefix=args.value_head_prefix,
            )
            get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    else:
        reward_model = None

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    if critic is not None:
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    if critic is not None:
        get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(actor)
    if critic is not None:
        strategy.print(critic)

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if critic is not None:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    critic_optim = None
    if critic is not None:
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)

    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
            train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )

    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
    num_update_steps_per_episodes = max(num_update_steps_per_episodes, 1) # ensure at least 1
    if args.custom_single_prompt:
        num_update_steps_per_episodes = args.max_epochs

    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        args.lr_scheduler,
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    critic_scheduler = None
    if critic_optim is not None:
        critic_scheduler = get_scheduler(
            args.lr_scheduler,
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

    if critic is not None:
        # prepare models/optimizers...
        (
            (actor, actor_optim, actor_scheduler),
            (critic, critic_optim, critic_scheduler),
            reward_model,
            initial_model,
        ) = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            (critic, critic_optim, critic_scheduler),
            reward_model,
            initial_model,
            is_rlhf=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    else:
        (
            (actor, actor_optim, actor_scheduler),
            reward_model,
            initial_model,
        ) = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            reward_model,
            initial_model,
            is_rlhf=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True,
                                     gradient_accumulation_steps=args.gradient_accumulation_steps)

    # load checkpoint
    consumed_samples = 0

    # if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
    if args.load_checkpoint and os.path.exists(f"{args.ckpt_path}_actor"):
        # _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"))
        _, states = strategy.load_ckpt(actor.model, f"{args.ckpt_path}_actor")
        if critic is not None:
            # strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))

            strategy.load_ckpt(critic, f"{args.ckpt_path}_critic")
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    if args.actor_learning_rate == 0:
        assert not args.shared_actorcritic # Should not do this with shared actor critic
        vf_coef = 100000 # Dummy value
    else:
        vf_coef = args.critic_learning_rate / args.actor_learning_rate

    true_posterior_samples = None
    if args.load_posterior_samples:

        print("Loading true posterior samples")

        true_posterior_samples_by_prompt_and_by_token = torch.load(f"{args.load_posterior_samples_name}")
        true_posterior_samples = \
            true_posterior_samples_by_prompt_and_by_token[
                0]
        true_posterior_samples = torch.tensor(
            true_posterior_samples,
            dtype=torch.int64)

        # print("--DEVICE CHECK--")
        # print(next(actor.parameters()).device)
        # print(true_posterior_samples.device)
        true_posterior_samples = true_posterior_samples.to(next(actor.parameters()).device)
        # print(true_posterior_samples.device)

    # configure Trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        target_dist_beta=args.target_dist_beta,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # remote reward model
        remote_rm_url=args.remote_rm_url,
        shared_actorcritic=args.shared_actorcritic,
        vf_coef=vf_coef,
        model_eval=args.model_eval,
        threshold=args.threshold,
        reward_cap=args.reward_cap,
        n_seeds_f_q=args.n_seeds_f_q,
        rm_type=args.rm_type,
        bc_coef=args.bc_coef,
        bc_steps=args.bc_steps,
        true_posterior_samples=true_posterior_samples,
        actor_loss_type=args.actor_loss_type,
        critic_loss_type=args.critic_loss_type,
        alpha=args.alpha
    )


    iwae_lbs_list, iwae_ubs_list, f_q_estimates_list, g_q_estimates_list = \
        trainer.fit(args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes, true_posterior_samples)


    print("FINAL RESULTS IWAE LB LIST", flush=True)
    print(iwae_lbs_list)
    print("FINAL RESULTS IWAE UB LIST", flush=True)
    print(iwae_ubs_list)
    print("FINAL RESULTS F_Q", flush=True)
    print(f_q_estimates_list)
    print("FINAL RESULTS G_Q", flush=True)
    print(g_q_estimates_list)

    print("SAVING RESULTS", flush=True)

    target_to_save = (
        f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list
    )

    info_name_str = get_info_name_str(args)
    save_str = f"{args.save_info_path}/f_q_g_q_iwae_bounds_OpenRLHF_{info_name_str}"
    torch.save(target_to_save, save_str)

    # ppo_trainer will do saving...
    # if args.save_actor:
    #     # save model checkpoint after fitting on only rank0
    #     strategy.save_model(
    #         ema_model if args.enable_ema else actor,
    #         tokenizer,
    #         args.save_path,
    #     )
    #
    # if args.save_value_network:
    #     strategy.save_model(
    #         critic,
    #         tokenizer,
    #         args.save_path + "_critic",
    #     )





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_actor", action="store_true", default=False)

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of PPO inner loop steps")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")

    parser.add_argument("--bc_coef", type=float, default=0.0, help="Do behaviour cloning on exact posterior samples (cheating for the sake of illustrating optimality)")
    parser.add_argument("--bc_steps", type=int, default=-1, help="Default -1 means always use bc_coef; otherwise, after bc_steps, set bc_coef to 0")

    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation. THIS DUPLICATION HAPPENS AT THE DATASET LEVEL"
    )
    parser.add_argument("--duplicate_rollout_batch_by", type=int, default=1, help="For each prompt in the rollout batch, copy this duplicate_rollout_batch_by many times, before rolling out/generating experience/sequences. Specifically used for twist learning, where we need multiple samples for approximate positive sampling. For twist learning methods, should be > 1")

    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    # parser.add_argument("--init_kl_coef", type=float, default=1., help="KL penalty in PPO") # DO NOT USE: USE THE TARGET_DIST_BETA
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")


    parser.add_argument("--alpha", type=float, default=0.5, help="Only for use in mixed_ctl_mse loss; choose how much to prioritize ctl")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument(
        "--custom_single_prompt", action="store_true", default=False, help="Use only a single custom prompt"
    )
    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "toxicity_threshold", "sentiment_threshold",
                                 "p_last_tokens", "toy_test", "toy_rlhf"])
    parser.add_argument("--threshold", type=float, default=-5., help="The threshold for the toxicity score")
    parser.add_argument("--reward_cap", type=float, default=10000, help="Only for use with toy_rlhf rm_type")


    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    parser.add_argument("--target_dist_beta", type=float, default=1., help="Beta in our SMC formulation of the target distribution of p_0 e^{beta r}")
    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_posterior_samples_name", type=str, default='.', help="Full filename of what to load for posterior samples")
    parser.add_argument("--save_info_path", type=str, default="./info")
    parser.add_argument("--n_samples_for_f_q", type=int, default=500, help="Number of samples to use for f_q")
    parser.add_argument("--n_seeds_f_q", type=int, default=4, help="Number of seeds to use for f_q")


    parser.add_argument("--update_steps_per_episode", type=int, default=1, help="Number of gradient updates (PPO loss outer loop) per episode")
    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")
    parser.add_argument("--no_test_info", action="store_true", help="don't do the f_q_g_q stuff")
    parser.add_argument("--test_info_every", type=int, default=1, help="Test info (e.g., F_q) after this many number of gradient updates")

    parser.add_argument("--actor_modulates_base", action="store_true", help="Use parameterization where actor outputs an addition (modulation) to base log prob")
    parser.add_argument("--shared_actorcritic", action="store_true", help="Use parameterization where actor and critic are just different heads, not separate networks. Uses actor lr for shared learning rate")
    parser.add_argument("--model_eval", action="store_true", help="Use model.eval() instead of model.train(). Turns off dropout and norm statistics")

    parser.add_argument("--no_critic", action="store_true", help="Do not use a critic")

    parser.add_argument("--clamp_reward", action="store_true", help="Clamp reward between -10 and 10")

    parser.add_argument("--additional_sd_divider", type=float, default=1., help="Reduce the SD on initialization of final linear layer (for CustomActor / --actor_modulates_base) further; additional divisor on SD")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps deepspeed config hyperparameter")

    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine_with_min_lr",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                 "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau",
                 "cosine_with_min_lr", "warmup_stable_decay"]
    )

    parser.add_argument(
        "--actor_loss_type", type=str, default="ppo",
        choices=[
            "ppo", "ctl", "sixo", "sixo_approxneg"
        ]
    )

    parser.add_argument(
        "--critic_loss_type", type=str, default="mse",
        choices=["mse", "ctl", "mixed_ctl_mse", "sixo", "sixo_approxneg"]
    )

    args = parser.parse_args()

    args.init_kl_coef = 1 / args.target_dist_beta
    print(f"Init KL coef set to: {args.init_kl_coef}, based on target_dist_beta {args.target_dist_beta}")
    # Because otherwise you don't have the equivalence between the RL formulation and the probabilistic inference formulation with target dist

    # assert args.init_kl_coef == 1 / args.target_dist_beta
    # assert args.target_dist_beta == args.init_kl_coef
    # if args.target_dist_beta != 1 or args.init_kl_coef != 1:
    #     raise Exception("Be very careful; this has not been tested; be careful that the PPO = inference formulation is correctly implemented. Be careful about x or 1/x also.")

    assert args.kl_target is None # Just use fixed KL for now

    if args.critic_pretrain is None:
        print("[Warning]: --critic_pretrain not specified, defaulting to --pretrain")
        args.critic_pretrain = args.pretrain
        # if not args.remote_rm_url:
        #     args.critic_pretrain = args.reward_pretrain
        # else:
        #     args.critic_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    assert not args.clamp_reward # TODO I have this as default no clamp everywhere; if you want clamp, modify the code for it
    if not args.custom_single_prompt:
        print("[Warning] stuff like clamp reward, and probably some other things changed, and not yet tested without custom_single_prompt. The rm_type stuff might also need to be modified too") # TODO

    if args.actor_loss_type != "ppo":
        assert args.actor_modulates_base # Need the twist formulation with the CustomActor for this
        args.no_critic = True # No (PPO) critic when using the twist formulation
        args.init_kl_coef = 0
        assert args.kl_target is None
        assert args.duplicate_rollout_batch_by > 1 # NOTE: this is also the "batch" or "number of particles" used in twist learning; for a given prompt, how many particles we use.


    train(args)
