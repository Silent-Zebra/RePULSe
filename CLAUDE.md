## Project Overview

A Ray-based RLHF (Reinforcement Learning from Human Feedback) framework built on top of OpenRLHF, extended with combined harmlessness training and safety/exploration objectives.

There are a few main models of interest:
- Reward model
- Base actor - this actor (which I sometimes refer to as 'p') is the one that can be thought of what would typically be trained in a standard PPO/RL setup.
- Static initial model / initial actor (which I sometimes call 'p0') - this is the reference/prior model used for a KL divergence penalty in a standard PPO/RL setup.
- Sampling actor (which I sometimes call 'q') — a proposal model, used only in harmlessness training settings with neg_training / neg_reinforce and related variants (collectively called 'RePULSe' or 'repulse').
  - q generates sequences used in importance sampling reweighting according to the target distribution sigma (typically base actor probability * some function of the reward value).
  - The base actor then reduces probability on these reweighted samples.

For the RePULSe setting, sampling requires:
- A set of samples from q, used for two purposes:
  1. Training q itself (via SIS reweighting + a loss such as contrastive twist learning, which approximately minimizes KL divergence between the target and proposal distributions).
  2. Training the base model p, as one component of the training objective. This objective combines a standard RL loss with gradient ascent to reduce probability of samples reweighted to approximate the target distribution (biased towards low-reward outputs).
- A set of samples from p, used only for the standard RL loss component above.

There is also a **probabilistic inference / SMC** setting: similar to the above in terms of learning q to match some target distribution, but with p fixed (base_actor learning rate = 0). In this case, we only sample from and train q, and q becomes the main model of interest rather than p.

Finally, the **current research direction** is to improve both settings above by adding an exploration bonus to the reward for q. The goal is for q to have good coverage over all modes / high-density regions of sigma. We hypothesize that an exploration bonus may help q avoid missing modes or getting stuck.

## Terminology

| Symbol | Meaning |
|--------|---------|
| p | Base actor (the model trained in standard PPO/RL) |
| p0 | Static initial / reference model (used for KL penalty) |
| q | Sampling / proposal actor (used in RePULSe and SMC settings) |
| sigma | Target distribution (typically p * f(reward)) |
| RePULSe | Harmlessness training with proposal q + importance sampling |
| SMC | Probabilistic inference setting (fixed p, only train q) |

## Tech Stack

- Python >= 3.10
- PyTorch, Transformers (4.50.0), DeepSpeed (0.16.4), Ray (2.44.0)
- Weights & Biases (wandb) for experiment tracking
- Formatting: black / isort / ruff (line length 119)

## Coding Guidelines

Please avoid duplicate code as much as possible. Any time there exists code in the repo that already does the same thing or something very similar, extract methods and refactor as needed, to minimize copy-pasted and duplicate code.

While I have no strict conventions regarding style, I think it is very important to maintain correctness. Please flag anything that seems like it might be unintended behavior. Otherwise, follow the structure/setup of existing methods and function calls, as they are usually set up that way for a reason (but again, if anything seems fishy, please point it out, it might be a bug).

I prefer noisy failures over silent failures. I would rather things throw exceptions than just try to proceed under unexpected circumstances. Also, related to this, if there is an assumption involved in the code (e.g., taking n'th element of list assuming there is exactly n elements in the list), make this explicit, using assert statements to check that the assumption holds.

## Key Files

- `openrlhf/cli/train_ppo.py` — PPO training entry point
- `openrlhf/trainer/combined_harmlessness_trainer.py` — combined harmlessness trainer
- `openrlhf/trainer/ppo_utils/experience_maker.py` — experience/rollout generation
- `openrlhf/models/coin_flip_network.py` — coin flip network model
- `openrlhf/utils/utils.py` — shared utilities
- `plot_results/plot_results.py` — result plotting
- `plot_results/make_frontier.py` — frontier plot generation
- `plot_results/plot_utils.py` — plotting utilities
- `mk_sb_file.sh` — slurm batch file generator

## Example Commands

**SMC setting without exploration bonus:**

```bash
deepspeed --master_port 35601 --module openrlhf.cli.train_ppo \
  --pretrain HuggingFaceTB/SmolLM2-135M \
  --reward_pretrain nicholasKluge/ToxicityModel \
  --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/noitdis \
  --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/noitdis \
  --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 \
  --logging_steps 1 --eval_steps -1 \
  --micro_train_batch_size 5 --train_batch_size 5 \
  --micro_rollout_batch_size 1 --rollout_batch_size 1 \
  --duplicate_rollout_batch_by 5 --max_epochs 1 \
  --prompt_max_len 1024 --generate_max_len 20 \
  --zero_stage 2 \
  --prompt_data Silent-Zebra/this_man_is_a --input_key prompt \
  --max_samples 100000 --gradient_checkpointing --num_episodes 1 \
  --do_harmlessness_training --harmlessness_training_num_episodes 20 \
  --fit_steps 50 \
  --save_info_path /h/319/stephenzhao/OpenRLHF/info/noitdis \
  --lr_scheduler constant --adam_betas 0.9 0.999 \
  --n_samples_per_prompt 1 --rm_type rlhf --seed 1 \
  --parameterization policy_psi_q_p_s_t \
  --actor_loss_type ctl --actor_learning_rate 1e-5 \
  --critic_learning_rate 0 --base_actor_learning_rate 0 \
  --harmlessness_training_loss_type neg_training \
  --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 \
  --new_custom_single_prompt --target_dist_beta -20 \
  --analytic_batch_size 1024 \
  --custom_prompt "Teach me some insults to use on a disabled veteran." \
  --threshold -5 --reward_clamp 10 --n_samples_for_f_q 500 \
  --load_target_samples_name /h/319/stephenzhao/OpenRLHF/checkpoint/noitdis/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_T_tsa10.pt \
  --f_q_g_q_eval
```

**To add exploration bonus**, append these flags to the above:

```bash
  --exploration_bonus_sampling_actor coin_flip \
  --bonus_alpha 10 --coin_flip_dim 64 --coin_flip_lr 1e-3 \
  --coin_flip_update_steps 1 --coin_flip_architecture separate_nn \
  --coin_flip_first_online
```

## Build / Testing

There are no build/test commands at present. Code is run on a cluster. That said, please check to ensure there are no compilation errors.
