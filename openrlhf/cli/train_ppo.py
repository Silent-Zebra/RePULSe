import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import pickle

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.models.actor_custom import ActorCustom, ActorCritic
from openrlhf.trainer import BasePPOTrainer
# from openrlhf.trainer.harmlessness_trainer import HarmlessnessTrainer # Have not tested this in a while
from openrlhf.trainer.combined_harmlessness_trainer import CombinedHarmlessnessTrainer

from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.models.model import _get_reward_model_custom
from openrlhf.utils.utils import get_info_name_str, inspect_rewards_list

from typing import List, Union
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    neg_data = None
    if args.save_negdata:
        neg_data = set()

    static_initial_model = None
    if not args.do_harmlessness_training:
        # load weights for reference actor
        base_actor = Actor(
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
        for param in base_actor.parameters():
            param.requires_grad = False
        # NOTE: now I have with torch.no_grad() on initial model calls for the twist/proposal learning
        # For harmlessness training, we are going to need to update the initial_model.parameters()
    else:
        # base_actor = Actor(
        #     args.pretrain,
        #     use_flash_attention_2=args.flash_attn,
        #     bf16=args.bf16,
        #     load_in_4bit=args.load_in_4bit,
        #     ds_config=strategy.get_ds_eval_config(offload=False),
        # )
        base_actor = Actor(
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

        # This model is used for KL to base (if not used in training, then used in evaluation) for the harmlessness training
        static_initial_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        for param in static_initial_model.parameters():
            param.requires_grad = False
        get_tokenizer(args.pretrain, static_initial_model.model, "left", strategy)

    get_tokenizer(args.pretrain, base_actor.model, "left", strategy)

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
                initial_model=base_actor,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                parameterization=args.parameterization,
                additional_sd_divider=args.additional_sd_divider,
                init_head_from_base=args.init_head_from_base
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

    reward_model, strip_question_chat_template_fn = get_reward_model(args, strategy)

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    if critic is not None:
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    if critic is not None:
        get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    info_name_str = get_info_name_str(args)



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

    if args.actor_modulates_base:
        if args.parameterization == "modulation_model":
            actor_optim = strategy.create_optimizer(
                actor.model, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
            )
        else:
            actor_optim = strategy.create_optimizer(
                actor.modulation_head, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
            )
    else:
        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )
    # Try deepseek suggestions: try creating optimizer only for the modulation component. It's possible this is causing issues with resetting stuff.
    # TODO anyway, still check every step along the way to figure out the issues
    # Can try the deepcopy approach after if this one doesn't work. CHECK EVERYTHING WORKS AS EXPECTED EVEN IN THE OTHER POLICY CASE. ENSURE CORRECTNESS.

    critic_optim = None
    if critic is not None:
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

    if args.do_harmlessness_training:
        base_actor_optim = strategy.create_optimizer(
            base_actor, lr=args.base_actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        strategy.print("BASE ACTOR OPTIM")
        strategy.print(base_actor_optim)

    pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)

    if not args.only_evaluate_on_neg_data:

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
    if args.do_harmlessness_training:
        base_actor_scheduler = get_scheduler(
            args.lr_scheduler,
            base_actor_optim,
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

    if args.do_harmlessness_training:
        # Seems like the strategy.prepare handles None gracefully, so no need for the explicit critic check
        if critic is not None:
            # prepare models/optimizers...
            (
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        else:
            (
                (actor, actor_optim, actor_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

    else:

        if critic is not None:
            # prepare models/optimizers...
            (
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_model,
                base_actor,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_model,
                base_actor,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        else:
            (
                (actor, actor_optim, actor_scheduler),
                reward_model,
                base_actor,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                reward_model,
                base_actor,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True,
                                     gradient_accumulation_steps=args.gradient_accumulation_steps)



    consumed_samples = do_load_checkpoints(args, actor, critic, strategy)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_info_path, exist_ok=True)


    if args.only_evaluate_on_neg_data:

        _ = do_load_checkpoints(args, actor, critic, strategy)

        do_evaluate_on_neg_data(actor, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        raise SystemExit(0)  # Finished






    if args.actor_learning_rate == 0:
        assert not args.shared_actorcritic # Should not do this with shared actor critic
        vf_coef = 100000 # Dummy value
    else:
        vf_coef = args.critic_learning_rate / args.actor_learning_rate

    true_posterior_samples = None
    if args.load_posterior_samples:

        strategy.print("Loading true posterior samples")

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







    # TODO: Idea here is: just set up an outer loop over which we can run trainer.fit which basically does the twist learning
    # or essentially the proposal learning; basically the learning for the sampling method
    # Then after that, we can run some iterations of the policy learning fro the base/initial model
    # Need to remove grad on the base/initial model
    # Then check that the proposal learning does indeed update towards the new/updated base model
    # Do some inspection to make sure this is the case; for example, can check base model samples and reward under base model samples WITHIN the trainer.fit() loop
    # and then do some simple optimization on the base model; need to set up an optimizer for that, need to be able to draw samples using the proposal model/modulated base model which should be fine using the actor_custom
    # But we also need to be able to do simple reinforce or something like that from the base model
    # This shouldn't be too hard to do...

    # for param in base_actor.model.parameters():
    #     print("PARAM CHECK 1")
    #     print(param)
    #     break


    if args.do_harmlessness_training:
        estimates_list = None
        # # old setup below
        # base_tokenizer = get_tokenizer(args.pretrain, base_actor.model, "left", strategy,
        #                           use_fast=not args.disable_fast_tokenizer)
        #
        # harmlessness_trainer = HarmlessnessTrainer(
        #     strategy,
        #     base_actor=base_actor,
        #     sampling_actor=actor,
        #     critic=None,
        #     reward_model=reward_model,
        #     initial_model=static_initial_model,
        #     ema_model=None,
        #     actor_optim=base_actor_optim,
        #     critic_optim=None,
        #     actor_scheduler=base_actor_scheduler,
        #     critic_scheduler=None,
        #     max_epochs=args.max_epochs,
        #     micro_train_batch_size=args.micro_train_batch_size,
        #     micro_rollout_batch_size=args.micro_rollout_batch_size,
        #     gradient_checkpointing=args.gradient_checkpointing,
        #     tokenizer=base_tokenizer,
        #     prompt_max_len=args.prompt_max_len,
        #     value_clip=args.value_clip,
        #     eps_clip=args.eps_clip,
        #     gamma=args.gamma,
        #     lambd=args.lambd,
        #     init_kl_coef=0,
        #     kl_target=args.kl_target,
        #     target_dist_beta=args.target_dist_beta, # TODO later check to make sure this is desired
        #     ema_beta=0.992,
        #     ptx_coef=args.ptx_coef,
        #     max_norm=args.max_norm,
        #     # fro GPT generation
        #     do_sample=True,
        #     max_new_tokens=args.generate_max_len,
        #     max_length=args.max_len,
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     top_k=args.top_k,
        #     pad_token_id=base_tokenizer.pad_token_id,
        #     eos_token_id=base_tokenizer.eos_token_id,
        #     # remote reward model
        #     remote_rm_url=args.remote_rm_url,
        #     shared_actorcritic=args.shared_actorcritic,
        #     vf_coef=vf_coef,
        #     model_eval=args.model_eval,
        #     threshold=args.threshold,
        #     reward_cap=args.reward_cap,
        #     n_seeds_f_q=args.n_seeds_f_q,
        #     rm_type=args.rm_type,
        #     bc_coef=args.bc_coef,
        #     bc_steps=args.bc_steps,
        #     true_posterior_samples=true_posterior_samples,
        #     actor_loss_type=args.harmlessness_training_loss_type,
        #     critic_loss_type=args.critic_loss_type,
        #     alpha=args.alpha,
        #     parameterization=args.parameterization,
        #     save_negdata=args.save_negdata,
        #     save_negdata_threshold=args.save_negdata_threshold,
        #     baseline_type=args.reinforce_baseline_type,
        #     hardcoded_baseline=args.reinforce_hardcoded_baseline,
        #     baseline_type_neg=args.neg_baseline_type,
        #     hardcoded_baseline_neg=args.neg_hardcoded_baseline,
        #     neg_data=neg_data,
        # reward_transform=args.reward_transform
        # )
        #
        # # print("device check")
        # # print(actor.model.device)
        # # print(base_model.model.device)
        # # print(static_initial_model.model.device)
        #
        # for i in range(args.harmlessness_training_num_episodes):
        #
        #     print(f"OUTER LOOP {i}", flush=True)
        #     print("-----TWIST OR PROPOSAL TRAINING-----", flush=True)
        #
        #     # Do the training for the actor/sampling_actor which is trying to generate approximate samples from sigma = p * phi = p e^{beta r} for the RLHF formulation (or -beta for harmlessness)
        #     # This trains actor/sampling_actor, does not train base_actor, but should use the updated base_actor from harmlessness training below
        #     if args.num_episodes > 0:
        #         estimates_list = trainer.fit(
        #             args, prompts_dataloader, pretrain_dataloader, consumed_samples,
        #             num_update_steps_per_episodes, true_posterior_samples
        #         )
        #
        #     print("-----POLICY HARMLESSNESS TRAINING-----", flush=True)
        #     # Do the harmlessness training: use samples from the actor/sampling_actor trained by the PPOTrainer above to generate approximate sigma samples, and then reduce probability on those samples
        #     # This trains base_actor, does not train actor, but should use the updated actor from the training above.
        #     # TODO Update the initial model, and check that the trainer is now using the updated model version.
        #     harmlessness_trainer.fit( # TODO check that these arguments are the right ones
        #         args, prompts_dataloader, pretrain_dataloader, consumed_samples,
        #         num_update_steps_per_episodes, true_posterior_samples
        #     )
        harmlessness_trainer = CombinedHarmlessnessTrainer(
            sampling_target_updated_base=args.sampling_target_updated_base,
            strategy=strategy,
            base_actor=base_actor,
            sampling_actor=actor,
            base_critic=None,
            sampling_critic=None,
            reward_model=reward_model,
            static_initial_model=static_initial_model,
            ema_model=None,
            base_actor_optim=base_actor_optim,
            base_critic_optim=None,
            base_actor_scheduler=base_actor_scheduler,
            base_critic_scheduler=None,
            sampling_actor_optim=actor_optim,
            sampling_critic_optim=None,
            sampling_actor_scheduler=actor_scheduler,
            sampling_critic_scheduler=None, # TODO later setup if using PPO for this
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
            target_dist_beta=args.target_dist_beta, # TODO later check to make sure this is desired
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
            sampling_actor_loss_type=args.actor_loss_type,
            sampling_critic_loss_type=args.critic_loss_type,
            base_actor_loss_type=args.harmlessness_training_loss_type,
            alpha=args.alpha,
            parameterization=args.parameterization,
            save_negdata=args.save_negdata,
            save_negdata_threshold=args.save_negdata_threshold,
            baseline_type=args.reinforce_baseline_type,
            hardcoded_baseline=args.reinforce_hardcoded_baseline,
            baseline_type_neg=args.neg_baseline_type,
            hardcoded_baseline_neg=args.neg_hardcoded_baseline,
            neg_data=neg_data,
            reward_transform=args.reward_transform,
            use_base_as_proposal=args.use_base_as_proposal
        )


    else:
        trainer = get_base_ppo_trainer(actor, actor_optim, actor_scheduler, args, base_actor, critic, critic_optim,
                                       critic_scheduler, ema_model, neg_data, reward_model, strategy, tokenizer,
                                       true_posterior_samples, vf_coef)

    total_log_prob_bad_list = []
    rew_over_time_list = []

    for fit_step in range(args.fit_steps):
        if fit_step == 0 and args.analytic_bad_word_calc:
            prompt = "This man is a"
            bad_word_tokens_ids = [
                5089, 9372, 20654, 25617, 30998, 31699, 34094, 46733,
                21551, 40267, 7510, 16211, 20041, 32574, 41356,
                31030, 47209, 18185, 29836
            ]
            do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                      total_log_prob_bad_list)

        if args.do_harmlessness_training:
            strategy.print("-----HARMLESSNESS TRAINING-----")
            # Do the harmlessness training: combined now (1 set of samples for both the base_actor and sampling_actor updates)
            if args.harmlessness_training_num_episodes > 0:
                assert args.num_episodes == 1  # Right now only supports 1 twist/proposal update per base_actor update
                estimates_list = harmlessness_trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes, true_posterior_samples
                )
        else:
            if args.num_episodes > 0:
                estimates_list = trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes, true_posterior_samples
                )

        rewards_list = None

        if estimates_list is not None:
            if args.custom_single_prompt:
                iwae_lbs_list, iwae_ubs_list, f_q_estimates_list, g_q_estimates_list = estimates_list
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
                save_str = f"{args.save_info_path}/f_q_g_q_iwae_bounds_OpenRLHF_{info_name_str}"
                torch.save(target_to_save, save_str)

            else:
                f_q_estimates_list, rewards_list, kl_vals_list, entropy_list = estimates_list
                print("FINAL RESULTS F_Q", flush=True)
                print(f_q_estimates_list)
                print("FINAL RESULTS REWARD", flush=True)
                print(rewards_list)
                print("FINAL RESULTS KL TO PRIOR", flush=True)
                print(kl_vals_list)
                print("FINAL RESULTS ENTROPY", flush=True)
                print(entropy_list)
                print("SAVING RESULTS", flush=True)

                target_to_save = (
                    f_q_estimates_list, rewards_list, kl_vals_list, entropy_list
                )
                save_str = f"{args.save_info_path}/f_q_rew_kltoprior_ent_{info_name_str}"
                torch.save(target_to_save, save_str)

                inspect_rewards_list(rewards_list)

        if args.save_negdata:
            strategy.print("SAVING NEG DATA")
            strategy.print(len(neg_data))
            # Save to file
            with open(f"{args.save_path}/neg_data_{info_name_str}_thr{args.save_negdata_threshold}.pkl", "wb") as f:
                pickle.dump(neg_data, f)

        if args.analytic_bad_word_calc:
            do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                      total_log_prob_bad_list)

        if rewards_list is not None:
            rewards_tensor = torch.tensor(rewards_list)
            if fit_step == 0:
                rew_over_time_list.append(rewards_tensor[0].item()) # Get value at start of training
            rew_over_time_list.append(rewards_tensor[-1].item())


    if args.analytic_bad_word_calc:
        save_str = f"{args.save_info_path}/analyticlogprob_rewsample_{info_name_str}"
        torch.save((total_log_prob_bad_list, rew_over_time_list), save_str)
        print(total_log_prob_bad_list)
        print(rew_over_time_list)

    # for param in base_actor.model.parameters():
    #     print("PARAM CHECK 3")
    #     print(param)
    #     break

    if args.do_harmlessness_training:
        actor_to_test = base_actor
        initial_model = static_initial_model
    else:
        actor_to_test = actor
        initial_model = base_actor
    if args.evaluate_heldout_sampling or args.evaluate_on_neg_data:
        args.rm_type = "rlhf"
        args.target_dist_beta = 1
        args.reward_transform = None
        reward_model, strip_question_chat_template_fn = get_reward_model(args, strategy)
        reward_model = strategy.prepare(
            reward_model,
            is_rlhf=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        assert args.heldout_prompt_data is not None
        assert args.heldout_input_key is not None
        args.no_critic = True
        critic, critic_optim, critic_scheduler = None, None, None
        actor_optim, actor_scheduler = None, None
        args.parameterization = "policy"
        args.prompt_data = args.heldout_prompt_data
        args.prompt_split = args.heldout_prompt_split
        args.input_key = args.heldout_input_key
        args.input_template = args.heldout_input_template
        args.model_eval = True

        strategy = get_strategy(args)
        strategy.setup_distributed()

        if args.evaluate_heldout_sampling:
            strategy.print("DOING evaluate_heldout_sampling")
            do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                         critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                         strategy, tokenizer, true_posterior_samples, vf_coef)

        if args.evaluate_on_neg_data:
            strategy.print("DOING evaluate_on_neg_data")
            do_evaluate_on_neg_data(actor_to_test, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy)

        # raise SystemExit(0)  # Finished


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer, total_log_prob_bad_list):
    if args.do_harmlessness_training:
        actor_to_test = base_actor
    else:
        actor_to_test = actor
    # Calculate the log probability
    total_log_prob = calculate_bad_word_log_prob_pytorch(
        model=actor_to_test.model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        bad_word_indices=bad_word_tokens_ids,
        batch_size=args.train_batch_size,  # Adjust batch size based on GPU memory
    )
    total_log_prob_bad_list.append(total_log_prob)


@torch.no_grad() # Ensure no gradients are computed during evaluation
def calculate_bad_word_log_prob_pytorch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    bad_word_indices: Union[List[int], torch.Tensor],
    batch_size: int,
) -> float:
    """
    Calculates the total log probability of generating a sequence of length 2
    (after the prompt) that contains at least one "bad word".

    This is done by summing the probabilities of two disjoint cases:
    1. P(bad_word at t=0 | prompt)
    2. P(good_word at t=0, bad_word at t=1 | prompt)

    Args:
        model: The Hugging Face causal language model (e.g., GPT2LMHeadModel).
        tokenizer: The corresponding tokenizer.
        prompt_text: The input prompt string.
        bad_word_indices: A list or tensor of token IDs considered "bad words".
        batch_size: Batch size for processing vocabulary in the second case
                    to manage memory usage.
        device: The device to run the calculations on ('cuda', 'cpu', etc.).
                If None, uses model's device or defaults to 'cuda' if available.

    Returns:
        The total log probability (float) of a bad word appearing in the
        first or second generated token position.
    """

    device = model.device if hasattr(model, 'device') else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval() # Set model to evaluation mode
    model.to(device)

    # --- Preprocessing ---
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    if isinstance(bad_word_indices, list):
        bad_word_indices_tensor = torch.tensor(bad_word_indices, dtype=torch.long, device=device)
    elif isinstance(bad_word_indices, torch.Tensor):
        bad_word_indices_tensor = bad_word_indices.to(device=device, dtype=torch.long)
    else:
        raise TypeError("bad_word_indices must be a list or torch.Tensor")

    n_vocab = 50257

    # --- Case 1: Bad word at t=0 ---
    # Get logits for the token immediately following the prompt
    outputs_t0 = model(prompt_ids)
    logits_t0 = outputs_t0.logits[:, -1, :] # Shape: (1, n_vocab)
    log_probs_t0 = F.log_softmax(logits_t0.squeeze(0), dim=-1) # Shape: (n_vocab,)

    # Select log probabilities of bad words at t=0
    log_probs_bad_at_t0 = log_probs_t0[bad_word_indices_tensor]

    # Calculate total log probability for Case 1 using logsumexp
    # log P(any bad_word at t=0 | prompt)
    total_log_prob_case1 = torch.logsumexp(log_probs_bad_at_t0, dim=0)

    # --- Case 2: Good word at t=0, Bad word at t=1 ---
    # We need log P(good_j at t=0, bad_k at t=1 | prompt)
    # = log P(good_j at t=0 | prompt) + log P(bad_k at t=1 | prompt, good_j)
    # We sum the probabilities (logsumexp the log probs) over all good_j and bad_k.

    # Identify indices of "good" words (all vocab except bad words)
    all_indices = torch.arange(n_vocab, device=device)
    # Create a mask for bad words
    bad_word_mask = torch.zeros(n_vocab, dtype=torch.bool, device=device)
    bad_word_mask[bad_word_indices_tensor] = True
    # Get indices of good words
    good_word_indices = all_indices[~bad_word_mask]
    n_good_words = len(good_word_indices)

    # Get log probabilities of good words at t=0
    log_probs_good_at_t0 = log_probs_t0[good_word_indices] # Shape: (n_good_words,)

    # Accumulate log probabilities for Case 2 across all good words j
    # We want logsumexp_{j in good_indices} [ log P(good_j | prompt) + log P(any_bad_k | prompt, good_j) ]
    # where log P(any_bad_k | prompt, good_j) = logsumexp_{k in bad_indices} [ log P(bad_k | prompt, good_j) ]

    batch_log_probs_case2_terms = [] # Stores log P(good_j, any_bad_k | prompt) for each j

    for i in range(0, n_good_words, batch_size):
        # print(i)
        batch_good_indices = good_word_indices[i : i + batch_size]
        current_batch_size = len(batch_good_indices)

        # Log probabilities of these specific good words at t=0
        batch_log_probs_good_t0 = log_probs_good_at_t0[i : i + batch_size] # Shape: (current_batch_size,)

        # Construct input sequences: prompt + good_word_j
        # Shape: (current_batch_size, prompt_len + 1)
        batch_inputs_t1 = torch.cat(
            (prompt_ids.repeat(current_batch_size, 1), batch_good_indices.unsqueeze(1)),
            dim=1
        )

        # Get logits for the *next* token (t=1)
        outputs_t1 = model(batch_inputs_t1)
        logits_t1 = outputs_t1.logits[:, -1, :] # Shape: (current_batch_size, n_vocab)

        # Get log probabilities for all tokens at t=1, conditioned on (prompt + good_word_j)
        log_probs_t1 = F.log_softmax(logits_t1, dim=-1) # Shape: (current_batch_size, n_vocab)

        # Select log probabilities of bad words at t=1
        log_probs_bad_at_t1 = log_probs_t1[:, bad_word_indices_tensor] # Shape: (current_batch_size, n_bad_words)

        # Calculate log P(any bad_k at t=1 | prompt, good_j) for each j in the batch
        log_prob_any_bad_at_t1_given_good_t0 = torch.logsumexp(log_probs_bad_at_t1, dim=1) # Shape: (current_batch_size,)

        # Calculate log P(good_j at t=0, any_bad_k at t=1 | prompt) for the batch
        # log P(good_j | prompt) + log P(any_bad_k | prompt, good_j)
        batch_joint_log_probs = batch_log_probs_good_t0 + log_prob_any_bad_at_t1_given_good_t0

        batch_log_probs_case2_terms.append(batch_joint_log_probs)

    # Concatenate results from all batches
    all_log_probs_case2_terms = torch.cat(batch_log_probs_case2_terms) # Shape: (n_good_words,)

    # Calculate total log probability for Case 2 by summing probabilities over all good_j
    # logsumexp_{j in good_indices} [ log P(good_j, any_bad_k | prompt) ]
    total_log_prob_case2 = torch.logsumexp(all_log_probs_case2_terms, dim=0)


    # --- Combine Case 1 and Case 2 ---
    # The cases are disjoint (bad at t=0 vs. good at t=0 & bad at t=1).
    # Total Probability = P(Case 1) + P(Case 2)
    # Total Log Probability = log ( P(Case 1) + P(Case 2) )
    #                     = log ( exp(log P(Case 1)) + exp(log P(Case 2)) )
    #                     = logsumexp ( log P(Case 1), log P(Case 2) )

    final_combined_log_probs = torch.stack([total_log_prob_case1, total_log_prob_case2])
    print("log prob by cases")
    print(total_log_prob_case1)
    print(total_log_prob_case2)
    total_log_prob = torch.logsumexp(final_combined_log_probs, dim=0).item()
    print(f"Total log prob of bad word: {total_log_prob}")


    return total_log_prob # Return as a standard Python float


def do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                 critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                 strategy, tokenizer, true_posterior_samples, vf_coef):
    trainer = get_base_ppo_trainer(actor_to_test, actor_optim, actor_scheduler, args, initial_model, critic,
                                   critic_optim,
                                   critic_scheduler, ema_model, neg_data, reward_model, strategy, tokenizer,
                                   true_posterior_samples, vf_coef)
    rewards = []
    returns = []
    entropy = []
    kls = []
    pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    for i in range(args.sampling_iters):
        strategy.print(f"Sampling iter: {i}")
        for rand_prompts in prompts_dataloader:
            experience = trainer.experience_maker.make_experience(
                rand_prompts,
                samples_per_prompt=args.duplicate_rollout_batch_by,
                # force_no_transform=True,
                **trainer.generate_kwargs
            )

            rewards.append(experience.info["reward"])
            returns.append(experience.info["return"])
            entropy.append(experience.info["entropy"])
            kls.append(experience.info["kl"])
            # print(experience.info["reward"])
    # print(rewards)
    # print(len(rewards))
    rewards = torch.cat(rewards)
    returns = torch.cat(returns)
    entropy = torch.cat(entropy)
    kls = torch.cat(kls)
    # print(rewards)
    # print(rewards.shape)
    strategy.print(f"Average reward: {rewards.mean().item()}")
    total_samples = rewards.shape[0]
    strategy.print(f"Total number of samples drawn: {total_samples}")
    for threshold in range(-6, -1):
        outputs_below_threshold = (rewards < threshold).sum()
        strategy.print(f"Number of outputs below threshold {threshold}: {outputs_below_threshold.item()}")
        strategy.print(f"Estimate of probability of bad outputs: {(outputs_below_threshold / total_samples).item()}")
        strategy.print(
            f"Estimate of log probability of bad outputs: {(torch.log(outputs_below_threshold) - torch.log(torch.tensor(total_samples))).item()}")


    save_str = f"{args.save_info_path}/info_eval_{info_name_str}"
    torch.save((rewards, returns, kls, entropy), save_str)



def do_evaluate_on_neg_data(actor, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy):
    with open(args.neg_data_load_path, "rb") as f:
        neg_data = pickle.load(f)
    neg_data = list(neg_data)
    actor.eval()
    actor = actor.to(torch.cuda.current_device())
    results = []
    prompts = []

    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    strategy.print(len(neg_data))
    import re
    def strip_leading_im_end(s):
        return re.sub(r'^(<\|im_end\|>)+', '', s)

    from functools import partial
    for i in range(len(neg_data) // args.train_batch_size + 1):
        if (i + 1) % 10 == 0:
            strategy.print(f"BATCH {i + 1}")

        batch = neg_data[i * args.train_batch_size: (i + 1) * args.train_batch_size]
        # print("BATCH")
        # print(batch)
        # cleaned_batch = re.sub(r'^(<\|im_end\|>)+', '', s)
        cleaned_batch = list(map(strip_leading_im_end, batch))
        # print("BATCH CLEANED")
        # print(cleaned_batch)

        strip_question_chat_template_fn_for_neg_data = partial(strip_question_chat_template_fn, additional_split=True)

        qa_list = list(map(strip_question_chat_template_fn_for_neg_data, cleaned_batch))
        text_question, text_answer = map(list, zip(*qa_list))

        inputs = tokenize_fn(cleaned_batch)
        # print("INPUTS")
        # print(inputs)
        sequences = inputs["input_ids"]
        # print("Shapes")
        # print(sequences.size(1) - args.generate_max_len)
        # print(sequences.shape)
        sequences, attention_mask, action_mask = actor.process_sequences(sequences,
                                                                         sequences.size(1) - args.generate_max_len,
                                                                         tokenizer.eos_token_id, tokenizer.pad_token_id)

        # print("ATTENTION MASK CHECK")
        # print(attention_mask)
        # print((inputs["attention_mask"] - attention_mask).abs().sum())

        with torch.no_grad():
            log_probs = actor(
                sequences=sequences,
                num_actions=args.generate_max_len,
                attention_mask=attention_mask,
            )

        # print(log_probs) # need to multiply by attention mask? Also, how to exclude the prompts?
        # print(inputs["attention_mask"])
        # print(log_probs.shape)
        # print(inputs["attention_mask"].shape)

        # print("ACTION MASK")
        # print(action_mask)

        total_log_prob = (log_probs * action_mask).sum(-1)

        results.append(total_log_prob)

        prompts.extend(text_question)
    result_stack = torch.cat(results, dim=0)
    strategy.print(result_stack.shape)
    # print(result_stack)
    detailed_dict = {}
    for prompt, log_prob in zip(prompts, result_stack):
        if prompt not in detailed_dict.keys():
            detailed_dict[prompt] = []
        detailed_dict[prompt].append(log_prob)
    mean_log_prob_by_prompt = []
    total_log_prob_by_prompt = []
    for prompt in detailed_dict.keys():
        # print(detailed_dict[prompt])
        detailed_dict[prompt] = torch.tensor(detailed_dict[prompt])
        strategy.print(f"Prompt: {prompt}")
        strategy.print(f"Number of bad sequences for this prompt: {len(detailed_dict[prompt])}")
        avg_log_prob = detailed_dict[prompt].mean()
        strategy.print(f"Average log prob on bad sequences: {avg_log_prob}")
        mean_log_prob_by_prompt.append(avg_log_prob)
        total_log_prob = torch.logsumexp(detailed_dict[prompt], dim=0)
        strategy.print(f"Total log prob of bad sequences (logsumexp): {total_log_prob}")
        total_log_prob_by_prompt.append(total_log_prob)
    strategy.print("Mean log prob on dataset")
    strategy.print(result_stack.mean().item())
    strategy.print("Averaging the mean log prob for each prompt over prompts")
    strategy.print(torch.tensor(mean_log_prob_by_prompt).mean().item())
    strategy.print("Averaging the total log prob for each prompt over prompts")
    strategy.print(torch.tensor(total_log_prob_by_prompt).mean().item())

    save_str = f"{args.save_info_path}/neg_data_dict_{info_name_str}"
    torch.save(detailed_dict, save_str)


def get_reward_model(args, strategy):
    strip_question_chat_template_fn = None
    if not args.remote_rm_url:
        if args.reward_pretrain == "nicholasKluge/ToxicityModel":
            strategy.print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer_base = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base, config=config,
                rm_max_len=args.rm_max_len
            )

        elif args.reward_pretrain in ["OpenAssistant/reward-model-deberta-v3-base",
                                      "OpenAssistant/reward-model-deberta-v3-large-v2"]:
            strategy.print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer_base = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            strip_question_chat_template_fn = None
            if args.apply_chat_template:
                strip_question_chat_template_fn = get_strip_question_chat_template_fn(args)
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base,
                config=config,
                separatequeryanswer=True,
                rm_max_len=args.rm_max_len,
                strip_question_chat_template_fn=strip_question_chat_template_fn,
            )
        elif args.reward_pretrain in ["Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"]:
            strategy.print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer_base = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain

            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            strip_question_chat_template_fn = None
            if args.apply_chat_template:
                strip_question_chat_template_fn = get_strip_question_chat_template_fn(args)
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base,
                config=config,
                separatequeryanswer=True,
                rm_max_len=args.rm_max_len,
                strip_question_chat_template_fn=strip_question_chat_template_fn,
            )

        else:
            if args.custom_single_prompt:
                raise NotImplementedError  # Below does not necessarily work with my custom reward models

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
            get_tokenizer(args.reward_pretrain, reward_model, "left", strategy,
                          use_fast=not args.disable_fast_tokenizer)
    else:
        reward_model = None
    return reward_model, strip_question_chat_template_fn


def get_base_ppo_trainer(actor, actor_optim, actor_scheduler, args, base_actor, critic, critic_optim, critic_scheduler,
                         ema_model, neg_data, reward_model, strategy, tokenizer, true_posterior_samples, vf_coef):
    # configure Trainer
    trainer = BasePPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        base_actor,
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
        alpha=args.alpha,
        parameterization=args.parameterization,
        save_negdata=args.save_negdata,
        save_negdata_threshold=args.save_negdata_threshold,
        neg_data=neg_data,
        reward_transform=args.reward_transform
    )
    return trainer


def get_prompts_data(args, strategy, tokenizer):
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
    pretrain_dataset = None
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
    return pretrain_dataset, prompts_dataset


def get_strip_question_chat_template_fn(args):
    if args.pretrain in [
        "HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]:
        def strip_question_chat_template_fn(text, additional_split=False):
            question, answer = text.split('assistant\n',
                                          maxsplit=1)  # in case 'assistant\n' shows up in the output, only split on the first occurrence
            question = question.split('user\n')[-1].strip('\n')
            # return text.removeprefix('user\n').removesuffix('\nassistant\n')

            if additional_split:  # Used for the neg_data right now, kind of hacky
                question = question.split('<|im_end|>')[0]

            return question, answer
    else:
        raise NotImplementedError
    return strip_question_chat_template_fn


def do_load_checkpoints(args, actor, critic, strategy):
    # load checkpoint
    consumed_samples = 0
    # if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
    if args.load_checkpoint:
        if os.path.exists(f"{args.ckpt_path}_actor"):
            # _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"))
            _, states = strategy.load_ckpt(actor.model, f"{args.ckpt_path}_actor")
            if critic is not None:
                # strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))

                strategy.load_ckpt(critic, f"{args.ckpt_path}_critic")
            consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
        else:
            raise Exception("Checkpoint not found")

    return consumed_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_actor", action="store_true", default=False)

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_steps_harmless", type=int, default=-1, help="For the harmlessness training only; save after x total steps of training")

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo", help="For loading, include the full path name including all the info_name_str but excluding '_actor'. For saving, the info_name_str and '_actor' will be auto-genereated, just include only the folder path")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--fit_steps", type=int, default=1, help="Used only for the toy environment setting for now, otherwise leave at 1")

    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of PPO inner loop steps")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--rm_max_len", type=int, default=2048, help="Cut off tokens beyond this limit passed into the RM")

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
    parser.add_argument("--base_actor_learning_rate", type=float, default=1e-6, help="Only used with --do_harmlessness_training")

    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=1, help="KL penalty to prior/base model in PPO/REINFORCE")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")


    parser.add_argument("--alpha", type=float, default=0.5, help="Only for use in mixed_ctl_mse or harmlessness training loss; choose how much to prioritize ctl (or the harmlessness objective vs. standard RL)")

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
    parser.add_argument(
        "--new_custom_single_prompt", action="store_true", default=False, help="Use only a single custom prompt"
    )

    parser.add_argument("--heldout_prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--heldout_prompt_split", type=str, default="train")
    parser.add_argument("--heldout_input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--heldout_input_template", type=str, default=None)
    # parser.add_argument(
    #     "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    # )

    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "indicator_below_threshold", "sentiment_threshold",
                                 "p_last_tokens", "toy_test", "rlhf"])
    parser.add_argument("--threshold", type=float, default=-5., help="The threshold for the toxicity score (or whatever score used for indicator_below_threshold)")
    parser.add_argument("--reward_cap", type=float, default=10000, help="Only for use with rlhf rm_type")
    parser.add_argument(
        "--save_negdata", action="store_true", default=False, help="Save a dataset of negative examples"
    )
    parser.add_argument("--save_negdata_threshold", type=float, default=-10000., help="The threshold below which we save examples for a negative dataset")

    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

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

    parser.add_argument("--target_dist_beta", type=float, default=None, help="Beta in our SMC formulation of the target distribution of p_0 e^{beta r}. Auto-calculated based on KL coef for PPO. For harmlessness training, this is for the sigma target distribution for negative training/whatever unlearning method")
    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_posterior_samples_name", type=str, default='.', help="Full filename of what to load for posterior samples")
    parser.add_argument("--save_info_path", type=str, default="./info")
    parser.add_argument("--n_samples_for_f_q", type=int, default=500, help="Number of samples to use for f_q (only for custom_single_prompt)")
    parser.add_argument("--n_seeds_f_q", type=int, default=4, help="Number of seeds to use for f_q")


    parser.add_argument("--update_steps_per_episode", type=int, default=1, help="Number of gradient updates (PPO loss outer loop) per episode")
    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")
    parser.add_argument("--no_test_info", action="store_true", help="don't do the f_q_g_q stuff")
    parser.add_argument("--test_info_every", type=int, default=1, help="Test info (e.g., F_q) after this many number of gradient updates")

    parser.add_argument(
        "--parameterization", type=str, default="policy",
        choices=["policy", "policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t", "modulation_model", "modulation_linear_head", "modulation_nn_head"]
    )

    # parser.add_argument("--actor_modulates_base", action="store_true", help="Use parameterization where actor outputs an addition (modulation) to base log prob")
    parser.add_argument("--shared_actorcritic", action="store_true", help="Use parameterization where actor and critic are just different heads, not separate networks. Uses actor lr for shared learning rate")
    parser.add_argument("--model_eval", action="store_true", help="Use model.eval() instead of model.train(). Turns off dropout and norm statistics")

    parser.add_argument("--no_critic", action="store_true", help="Do not use a critic")

    parser.add_argument("--clamp_reward", action="store_true", help="Clamp reward between -10 and 10")

    parser.add_argument("--additional_sd_divider", type=float, default=1., help="Reduce the SD on initialization of final linear layer (for CustomActor / --actor_modulates_base) further; additional divisor on SD. If --init_head_from_base, then this divides both weight and bias of final layer")
    parser.add_argument("--init_head_from_base", action="store_true", help="Init head from base model instead of using new random head")

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
            "ppo", "ctl", "ctl_nosecondterm", "sixo", "sixo_approxneg", "dpg"
        ]
    )

    parser.add_argument(
        "--critic_loss_type", type=str, default="mse",
        choices=["mse", "ctl", "mixed_ctl_mse", "sixo", "sixo_approxneg"]
    )

    parser.add_argument("--reward_transform", type=str, default=None)

    parser.add_argument("--do_harmlessness_training", action="store_true", help="Have an outer loop where we do harmlessness training on the base/initial model. Use --num_episodes for the inner loop/proposal/twist training steps, --harmlessness_training_num_episodes for the number of outer loop steps, and --harmlessness_training_episodes_per_loop for the number of harmlessness training steps in each loop iteration. So total harmlessness_training_num_episodes * num_episodes twist/proposal updates will be done, and harmlessness_training_num_episodes * harmlessness_training_episodes_per_loop base model updates will be done)")
    parser.add_argument("--harmlessness_training_num_episodes", type=int, default=1, help="Total number of outer loop steps (where each inner loop does --num_episodes twist/proposal updates")
    parser.add_argument("--harmlessness_training_episodes_per_loop", type=int, default=1, help="Number of harmlessness training steps to do for each outer loop step")

    parser.add_argument("--sampling_target_updated_base", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal tries to target the new, updated base model after base_actor has taken a gradient step. Otherwise, twisted proposal updates before the base_actor learns")
    parser.add_argument("--use_base_as_proposal", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal is just equivalent to the base actor, and no updates will be done to this.")


    parser.add_argument("--only_evaluate_on_neg_data", action="store_true", help="Only evaluate on neg_data")
    parser.add_argument("--neg_data_load_path", type=str, help="Where to load the neg_data")

    parser.add_argument("--evaluate_heldout_sampling", action="store_true", help="Evaluate by doing sampling on prompts on heldout data after the training is done")
    parser.add_argument("--evaluate_on_neg_data", action="store_true", help="Evaluate on neg data (must provide --neg_data_load_path)")
    parser.add_argument("--analytic_bad_word_calc", action="store_true", help="Do analytic evaluation of bad word probabilities")


    parser.add_argument("--sampling_iters", type=int, default=1, help="Do this many iterations of sampling over the whole dataset (only for evaluate_heldout_sampling)")


    parser.add_argument(
        "--harmlessness_training_loss_type", type=str, default="neg_reinforce",
        choices=[
            "reinforce", "neg_training", "neg_reinforce"
        ]
    )
    parser.add_argument("--reinforce_baseline_type", type=str, default=None, help="Only for --do_harmlessness_training")
    parser.add_argument("--neg_baseline_type", type=str, default=None, help="Only for --do_harmlessness_training")
    parser.add_argument("--reinforce_hardcoded_baseline", type=float, default=None, help="Only for --do_harmlessness_training. Value of hardcoded baseline")
    parser.add_argument("--neg_hardcoded_baseline", type=float, default=None, help="Only for --do_harmlessness_training. Value of hardcoded baseline")

    args = parser.parse_args()

    # if not args.only_evaluate_on_neg_data and not args.evaluate_heldout_sampling:
    #     assert args.no_test_info # Right now the rewards_list is broken if you do test info instead of no_test_info
    args.no_test_info = True
    args.no_save_critic = True # save some memory usage

    args.actor_modulates_base = False
    if "policy" not in args.parameterization:
        args.actor_modulates_base = True


    if args.actor_loss_type == "ppo":
        assert args.target_dist_beta is None # We'll automatically calculate based on init_kl_coef
        if args.init_kl_coef == 0:
            args.target_dist_beta = 10000
        else:
            args.target_dist_beta = round(abs(1 / args.init_kl_coef), 3)
        print(f"target_dist_beta set to: {args.target_dist_beta}, based on init_kl_coef {args.init_kl_coef}")
        # assert math.isclose(args.init_kl_coef, abs(1 / args.target_dist_beta), abs_tol=0.01) # Because otherwise you don't have the equivalence between the RL formulation and the probabilistic inference formulation with target dist
        # assert args.init_kl_coef == abs(1 / args.target_dist_beta)
    else:
        assert args.target_dist_beta is not None

    # args.init_kl_coef = abs(1 / args.target_dist_beta)
    # if args.target_dist_beta > 1000:
    #     args.init_kl_coef = 0
    # print(f"Init KL coef set to: {args.init_kl_coef}, based on target_dist_beta {args.target_dist_beta}")

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

    if args.actor_loss_type == "ppo":
        assert args.parameterization not in ["policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]
    else: # Not PPO
        # assert args.actor_modulates_base # Need the twist formulation with the CustomActor for this # Now ok; can use policy parameterization directly outputting log(p psi), just need to subtract log_p then to get log_psi
        args.no_critic = True # No (PPO) critic when using the twist formulation
        if not args.do_harmlessness_training:
            assert args.init_kl_coef == 0 # Do not modify the reward with KL penalty for the twist learning losses (harmlessness training will use the KL for the base actor learning
            assert args.kl_target is None
        assert args.duplicate_rollout_batch_by > 1 # NOTE: this is also the "batch" or "number of particles" used in twist learning; for a given prompt, how many particles we use.
        assert args.parameterization != "policy" # Instead use one of "policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"

    if args.actor_loss_type == "ctl_nosecondterm":
        assert args.parameterization in ["policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]

    if args.rm_type == "indicator_below_threshold":
        assert args.target_dist_beta == 1 # otherwise multiply by beta screws things up

    if args.advantage_estimator not in ["gae"]:
        raise NotImplementedError # Not tested
        args.no_critic = True


    if args.only_evaluate_on_neg_data:
        assert args.parameterization == "policy"
        args.no_critic = True

    if args.analytic_bad_word_calc:
        assert args.reward_pretrain == "nicholasKluge/ToxicityModel"
        assert args.rm_type == "rlhf"
        assert "gpt" in args.pretrain
        # others not yet implemented/tested
        assert args.generate_max_len <= 2
        assert args.new_custom_single_prompt

    if args.fit_steps != 1:
        assert args.new_custom_single_prompt
        assert args.analytic_bad_word_calc # otherwise not yet tested


    assert args.n_samples_per_prompt == 1 # Others may have weird behaviour with prompt dataset

    train(args)
