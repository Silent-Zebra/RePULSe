# Improving Coverage in Probabilistic Inference for Language Models

This is the codebase for "Improving Coverage in Probabilistic Inference for Language Models".

Paper link: TBD

# Setup Notes:

Will vary depending on your setup/cluster. When running commands, can avoid using the --adam_offload flag, if you run into issues with building DeepSpeedCPUAdam.

## Example commands for Vector cluster

On cluster: first consider deleting cache if the below commands don't work: ```rm -rf ~/.cache```

Then run the following commands (this does setup on a GPU which is useful for linking certain things):
```
srun -c 4 --gres=gpu:1 --mem=10GB -p a40 --pty bash
cd ~/OpenRLHF/
/pkgs/python-3.10.12/bin/python3 -m venv ~/OpenRLHF/newenv
source newenv/bin/activate
module load cuda-12.3
pip install -e .
pip install vllm
pip install flash-attn --no-build-isolation
```
Check that the installation works as expected:
```
python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

## Example commands for Compute Canada cluster:

```
rm -rf ~/.cache

module --force purge

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load cuda/12.6
module load scipy-stack/2024a
module load gcc arrow/18.1.0 
module load opencv/4.12.0
module load rust

virtualenv --no-download ENV

source ENV/bin/activate

pip install --no-index --upgrade pip

pip install --no-index torch deepspeed

pip install -r requirements.txt --no-index

pip install flash-attn --no-build-isolation
```

## Example commands for DCS cluster (using conda):

```
source /pkgs/anaconda310/etc/profile.d/conda.sh
conda create -n openrlhf python=3.10 -y
conda activate openrlhf
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install deepspeed
pip install -e .
```


# Commands Used in Experiments

## Toy Experiments

Below I provide the deepspeed training commands, although these were auto-generated using my scripts. To use the sbatch-generating scripts, use commands like:
`
bash mk_sb_file.sh --cluster default $x
`
where $x should be replaced with a full deepspeed command excluding "deepspeed --master_port xxxxx". Replace the arguments for --save_path, --ckpt_path, --save_info_path, --load_target_samples_name, --heldout_target_samples_name, with your folder paths and file names. Of course, you'd need to change the setup in mk_sb_file.sh in order to fit the specifics of your cluster.

You may then use 
`
bash mk_sb_files_seeds_2_to_x.sh 10 $x
`
where $x should be the generated sbatch file, to generate sbatch files for seeds 2 to 10 for the same setting.

### Setting 1

Baseline command:
```
deepspeed --master_port 36931 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinflen1remodevpos --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinflen1remodevpos --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 5 --train_batch_size 5 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 1 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 20 --fit_steps 100 --save_info_path /h/319/stephenzhao/OpenRLHF/info/probinflen1remodevpos --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-5 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --analytic_calc --new_custom_single_prompt --target_dist_beta 10 --analytic_batch_size 1024 --custom_prompt "This man is a"
```

For tempering, add:
```--anneal_target_dist_beta --start_target_dist_beta 1```

For CFN, add:
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 1 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0```




#### REINFORCE

```
deepspeed --master_port 39845 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt
```


#### REINFORCE with reward transformation

```
deepspeed --master_port 36595 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt --reward_transform minus_alpha_exp_beta_r
```


#### Base model proposal

```
deepspeed --master_port 33215 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.01 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt --use_base_as_proposal
```


#### RePULSe

```
deepspeed --master_port 38115 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 5 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 5 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.01 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt
```

### Example commands with 10 KL penalty (Appendix figure)

#### REINFORCE

```
deepspeed --master_port 35171 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta 0 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-5 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 10 --analytic_bad_word_calc --new_custom_single_prompt
```

#### REINFORCE with reward transformation

```
deepspeed --master_port 35861 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-5 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --init_kl_coef 10 --analytic_bad_word_calc --new_custom_single_prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 33631 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-5 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 10 --init_kl_coef 10 --analytic_bad_word_calc --new_custom_single_prompt --reward_transform minus_alpha_exp_beta_r
```

#### RePULSe

```
deepspeed --master_port 37441 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 5 --fit_steps 50 --target_dist_beta -1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 100 --init_kl_coef 10 --analytic_bad_word_calc --new_custom_single_prompt
```
```
deepspeed --master_port 34881 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 5 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 100 --init_kl_coef 10 --analytic_bad_word_calc --new_custom_single_prompt
```

## More Realistic Experiment (Sec 4.3)

General workflow: first run the training commands below. Then run GCG evaluation.

Below I provide the deepspeed training commands, although these were auto-generated using my scripts. To use the sbatch-generating scripts, use commands like:
`
bash mk_sb_file_seed1_dcs.sh $x
`
or 
`
bash mk_sb_file_multinode_seed1.sh $x
`
for multinode/distributed training experiments, where $x should be replaced with a full deepspeed command excluding "deepspeed --master_port xxxxx". Replace paths --save_path, --ckpt_path, --save_info_path, with your folder paths. Note that multinode training introduces nondeterminism, such that even setting the seed does not guarantee exact reproducibility. There are ways to improve determinism but this typically comes at a cost to efficiency/speed, so I avoid making that tradeoff. Unfortunately, this means that if you run my exact commands (for the 1B model experiments), you may get different results. You should get (almost) exactly the same results for the smaller models. 

You may then use `
bash mk_sb_files_seeds_2_to_x.sh 10 $x
`
where $x should be the generated sbatch file, to generate sbatch files for seeds 2 to 10 for the same setting.

For the Appendix experiments (same number of updates for the LM p), replace `--harmlessness_training_num_episodes 4` with `--harmlessness_training_num_episodes 2`, or `--num_episodes 4` with `--num_episodes 2` for PPO.

You may change `--heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1`, for example, to `--heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_2`, for val/test split from a similar data source.

### Setting 1: SmolLM-135M-Instruct as the LM and Deberta-v3-large-v2 as the RM

#### PPO

```
deepspeed --master_port 32161 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 4 --init_kl_coef 0.2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl20930 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy --actor_loss_type ppo --actor_learning_rate 3e-5 --critic_learning_rate 3e-5 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```


#### REINFORCE

```
deepspeed --master_port 35881 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta 0 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```

#### REINFORCE with Reward Transformation

```
deepspeed --master_port 34631 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 37981 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.5 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0.3 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 37801 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.3 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 38541 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.5 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```

#### Base model proposal

```
deepspeed --master_port 30811 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -30 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 0.2 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --rm_max_len 300 --use_base_as_proposal
```
```
deepspeed --master_port 39431 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -30 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 1 --init_kl_coef 0.2 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --rm_max_len 300 --use_base_as_proposal
```

#### RePULSe

```
deepspeed --master_port 38501 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-5 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 0.2 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --rm_max_len 300
```
```
deepspeed --master_port 30081 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --target_dist_beta -20 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 2e-5 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 0.2 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --rm_max_len 300
```
```
deepspeed --master_port 36191 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --target_dist_beta -30 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 1e-5 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 0.2 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --rm_max_len 300
```

### Setting 2: Llama-3.2-1B-Instruct as the LM and Skywork-Reward-V2-Llama-3.2-1B as the RM

#### PPO

```
deepspeed --master_port 38711 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 4 --init_kl_coef 2 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy --actor_loss_type ppo --actor_learning_rate 3e-7 --critic_learning_rate 3e-6 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```


#### REINFORCE

```
deepspeed --master_port 30441 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta 0 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-7 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --init_kl_coef 2 --alpha 0 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```

#### REINFORCE with Reward Transformation

```
deepspeed --master_port 33251 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20v2 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20v2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.3 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20v2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-7 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --init_kl_coef 2 --alpha 3 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 37401 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -1 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-7 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --init_kl_coef 2 --alpha 1 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 39601 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.3 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-7 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --init_kl_coef 2 --alpha 3 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 38111 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -1 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-7 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --init_kl_coef 2 --alpha 1 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```

#### Base model proposal

```
deepspeed --master_port 34701 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -10 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-7 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 2 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --use_base_as_proposal
```
```
deepspeed --master_port 34341 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -10 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-7 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 1 --init_kl_coef 2 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --use_base_as_proposal
```

#### RePULSe

```
deepspeed --master_port 39691 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --target_dist_beta -5 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-7 --critic_learning_rate 0 --base_actor_learning_rate 3e-7 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.1 --init_kl_coef 2 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```
```
deepspeed --master_port 37051 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --target_dist_beta -5 --save_info_path /scratch/zhaostep/OpenRLHF/info/rlhfmultikl20 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-7 --critic_learning_rate 0 --base_actor_learning_rate 3e-7 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.2 --init_kl_coef 2 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```


### Example GCG Evaluation Command

```
deepspeed --master_port 32061 --module openrlhf.cli.evaluate_gcg_sz     --pretrain "HuggingFaceTB/SmolLM-135M-Instruct"     --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor     --file_path data/harmful_behaviors_custom.csv     --atk_success_criteria reward     --reward_pretrain "OpenAssistant/reward-model-deberta-v3-large-v2"     --reward_threshold -4.0     --rm_exclude_adv_suffix          --zero_stage 2     --gradient_checkpointing          --parameterization policy     --max_targets 100     --max_new_tokens 20     --scenario behaviors     --gcg_steps 250     --gcg_search_width 512     --gcg_topk 256     --gcg_batch_size 512     --gcg_n_replace 1     --gcg_buffer_size 0     --gcg_use_prefix_cache     --gcg_filter_ids
```

Replace the --ckpt_path with whatever was saved from the previous training commands.

Sbatch file with the above command can be created using:

```
bash mk_eval_gcg_file_dcs.sh 250 20 /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor HuggingFaceTB/SmolLM-135M-Instruct OpenAssistant/reward-model-deberta-v3-large-v2 -4.0  harmful_behaviors_custom.csv
```

Yes, threshold -4.0 is correct here. I set this up in a kind of stupid way where for whatever threshold x you pass in, I calculate the samples with reward < x and with reward < x-1, and then my plotting code takes the x-1 results. A smarter way would be to just save the rewards of samples, and then dynamically set the threshold when plotting. I have this setup for the main frontier results (probability of bad output/CVaR) but didn't have time to change it for the GCG attacks yet.

## Plotting Results

To generate the plots of training over time (Sec 4.2), use: 
```
python plot_results/plot_results.py
```
The main thing to change in the plot_results.py file is "figname_modifier", to choose what plot to build. Of course, if you rerun my commands with different settings and want to plot those, you'd have to modify the "labels" and "load_prefixes_to_use".

To generate the frontiers (Sec 4.3), use:
```
python plot_results/make_frontier.py
```
Again, "figname_modifier" is the main thing to change; use "cvar" in the modifier if you want CVaR on the y-axis instead, use "gcg" if you want to plot the GCG attack success rate. If you use your own runs, you need to modify "labels", "load_prefixes_to_use", and "gcg_prefixes" for GCG attacks. 


# NOTE: This is built on top of the [RePULSe repo](https://github.com/Silent-Zebra/RePULSe), which itself is a fork of the [OpenRLHF repo](https://github.com/OpenRLHF/OpenRLHF).

