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

## Build / Testing

There are no build/test commands at present. Code is run on a cluster. That said, please check to ensure there are no compilation errors.
