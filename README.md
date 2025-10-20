# RePULSe 

TODO: Update with link to paper (Arxiv?), and brief description of the paper

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


# Commands Used in Experiments

## Toy Experiment (Sec 4.2)

Below I provide the deepspeed training commands, although these were auto-generated using my scripts. To use the sbatch-generating scripts, use commands like:
`
bash mk_sb_file_seed1_dcs_simple.sh $x
`
where $x should be replaced with a full deepspeed command excluding "deepspeed --master_port xxxxx". Replace paths --save_path, --ckpt_path, --save_info_path, with your folder paths.

You may then use 
`
bash mk_sb_files_seeds_2_to_x.sh 5 $x
`
where $x should be the generated sbatch file, to generate sbatch files for seeds 2 to 5 for the same setting.

### Example commands with 0 KL penalty (main paper figure)

#### PPO

```
deepspeed --master_port 39225 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --adam_offload --gradient_checkpointing --num_episodes 10 --fit_steps 50 --init_kl_coef 0 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy --actor_loss_type ppo --actor_learning_rate 1e-4 --critic_learning_rate 3e-5 --analytic_bad_word_calc --new_custom_single_prompt
```

#### REINFORCE

```
deepspeed --master_port 39845 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --adam_offload --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt
```


#### REINFORCE with reward transformation

```
deepspeed --master_port 36595 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --adam_offload --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt --reward_transform minus_alpha_exp_beta_r
```


#### Base model proposal

```
deepspeed --master_port 33215 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --adam_offload --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 10 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.01 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt --use_base_as_proposal
```


#### RePULSe

```
deepspeed --master_port 38115 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/toyrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 500 --train_batch_size 500 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 500 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 2 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --adam_offload --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 5 --fit_steps 50 --target_dist_beta -10 --save_info_path /h/319/stephenzhao/OpenRLHF/info/toyrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 5 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-4 --critic_learning_rate 0 --base_actor_learning_rate 1e-4 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0.01 --init_kl_coef 0 --analytic_bad_word_calc --new_custom_single_prompt
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
for multinode/distributed training experiments, where $x should be replaced with a full deepspeed command excluding "deepspeed --master_port xxxxx". Replace paths --save_path, --ckpt_path, --save_info_path, with your folder paths.

You may then use `
bash mk_sb_files_seeds_2_to_x.sh 10 $x
`
where $x should be the generated sbatch file, to generate sbatch files for seeds 2 to 10 for the same setting.

For the Appendix experiments (same number of updates for the LM p), replace `--harmlessness_training_num_episodes 4` with `--harmlessness_training_num_episodes 2`, or `--num_episodes 4` with `--num_episodes 2` for PPO.

You may change `--heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1`, for example, to `--heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_2`, for val/test split from a similar data source.

### Setting 1: SmolLM-135M-Instruct as the LM and Deberta-v3-large-v2 as the RM

#### PPO

```
deepspeed --master_port 32161 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 4 --init_kl_coef 0.2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl20930 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy --actor_loss_type ppo --actor_learning_rate 3e-5 --critic_learning_rate 3e-5 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```


#### REINFORCE

```
deepspeed --master_port 35881 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta 0 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt
```

#### REINFORCE with Reward Transformation

```
deepspeed --master_port 34631 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 37981 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.5 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 0.3 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 37801 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.3 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
```
```
deepspeed --master_port 38541 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --ckpt_path /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2 --save_steps -1 --save_steps_harmless 100 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 4 --target_dist_beta -0.5 --save_info_path /h/319/stephenzhao/OpenRLHF/info/rlhfmultilen20kl2 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --test_info_every 100 --n_seeds_f_q 1 --no_test_info --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 3e-5 --init_kl_coef 0.2 --harmlessness_training_loss_type reinforce --reinforce_baseline_type expectation --alpha 1 --save_negdata --save_negdata_threshold -5 --evaluate_heldout_sampling --sampling_iters 1 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_input_key prompt --reward_transform minus_alpha_exp_beta_r
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


# NOTE: Since this is a fork of the OpenRLHF repo, most of the commands are built on top of the OpenRLHF pipeline (but since this was forked many months ago, things are now slightly outdated, and I have not merged all of the newest OpenRLHF changes into this repo). The rest of the below is from the original OpenRLHF repo when it was forked.



<div align="center">
    <img alt="OpenRLHF logo" src="./docs/logo.png" style="height: 140px;" />
</div>
<div align="center">
<p align="center">
      <a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/OpenRLHF/OpenRLHF" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/OpenRLHF/OpenRLHF?color=0088ff" />
      <a href="https://github.com/OpenRLHF/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

<span>[ English | <a href="README_zh.md">中文</a> | <a href="README_ja.md">日本語</a> ]</span>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, and seamlessly compatible with Huggingface models and datasets.
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Packing Samples and vLLM generation acceleration, the performance of OpenRLHF 3~4x+ that of Optimized DeepSpeedChat with Hybrid Engine.
- **Distributed RLHF**:  OpenRLHF distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 70B+ models with multiple A100 80G GPUs and vLLM and 7B models across multiple 24GB RTX 4090 GPUs.
- **Hybrid Engine**:  OpenRLHF also supports the hybrid engine, allowing all models and vLLM engines to share the GPUs to avoid GPU idling.
- **PPO Implementation Optimization**: We integrated the implementation tricks for PPO to improve the training stability, referencing [Zhihu](https://zhuanlan.zhihu.com/p/622134699) and [Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361).

More details are in [Slides](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [Technical Report](https://arxiv.org/abs/2405.11143) | [Documents](https://openrlhf.readthedocs.io/)

## News
- [2025/3] The CMU [Advanced Natural Language Processing Spring 2025](https://cmu-l3.github.io/anlp-spring2025/) course uses OpenRLHF as the RLHF framework teaching case.
- [2025/2] [Logic-RL](https://arxiv.org/abs/2502.14768) and [PRIME](https://arxiv.org/abs/2502.01456) demonstrate that REINFORCE++ is more stable in training compared to GRPO and faster than PPO.
- [2025/2] StepFunc implements a [single-controller version of OpenRLHF](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero).
- [2025/2] [LMM-R1](https://github.com/TideDra/lmm-r1) is a fork of OpenRLHF, aimed at providing high-performance RL infrastructure for reproduction of DeepSeek-R1 on multimodal tasks.
- [2025/2] MIT & Microsoft proposed the [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) using OpenRLHF
- [2025/1] HKUST reproduced the [DeepSeek-R1-Zero and DeepSeek-R1 training on small models using OpenRLHF](https://github.com/hkust-nlp/simpleRL-reason)
- [2024/12] We "proposed" 😊 the [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://www.researchgate.net/publication/387487679_REINFORCE_A_SIMPLE_AND_EFFICIENT_APPROACH_FOR_ALIGNING_LARGE_LANGUAGE_MODELS).
- [2024/12] We analyzed the PPO, REINFORCE++, GRPO and RLOO in the [Notion Blogpost](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05).


## Features

- Distributed [PPO](./examples/scripts/train_ppo_llama_ray.sh) and [REINFORCE++/REINFORCE++-baseline/GRPO/RLOO](./examples/scripts/train_reinforce_llama_ray.sh) implementations based on Ray.  
- [Ray-based Reinforced Finetuning](./examples/scripts/train_ppo_llama_with_reward_fn.sh)
- Support Ray-based [PPO](./examples/scripts/train_ppo_llama_ray_hybrid_engine.sh) and [REINFORCE++/REINFORCE++-baseline/GRPO/RLOO](./examples/scripts/train_reinforce_llama_ray_hybrid_engine.sh) using Hybrid Engine  (`--colocate_all_models`, `--vllm_enable_sleep` and `--vllm_gpu_memory_utilization 0.5`)
- Full RLHF fine-tuning support for models with [over 70 billion parameters](./examples/scripts/train_ppo_llama_ray_70b.sh).  
- Integration with vLLM for accelerated generation in RLHF tasks (`--vllm_num_engines`).  
- Support for multiple reward models (`--reward_pretrain model1,model2...`) and remote reward models (`--remote_rm_url`).  
- Implementation of [DPO (Direct Preference Optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh) and [Kahneman-Tversky Optimization (KTO)](./examples/scripts/train_kto_llama.sh).  
- Support for [Iterative DPO](./examples/scripts/train_iterative_dpo_llama.sh) ([GitHub: Online-RLHF](https://github.com/RLHFlow/Online-RLHF)).  
- Support for [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).  
- Implementation of [Conditional SFT](./examples/scripts/train_conditional_llama.sh) ([arXiv:2308.12050](https://arxiv.org/abs/2308.12050)).  
- Support for [Knowledge Distillation](./examples/scripts/train_knowledge_distillation.sh) ([Microsoft: minillm](https://github.com/microsoft/LMOps/tree/main/minillm)).  
- Integration of [Process Reward Model (PRM)](./examples/scripts/train_prm_mistral.sh).  
- Packing of training samples for SFT, DPO, RM, PRM, and PPO (`--packing_samples`).  
- Implementation of [RingAttention](./examples/scripts/train_dpo_ring_llama.sh) (`--ring_attn_size`, `--ring_head_stride`).  
- Support for [Mixture of Experts (MoE)](./examples/test_scripts/train_sft_mixtral_lora.sh) (`--aux_loss_coef`).  
- Integration of FlashAttention2 (`--flash_attn`).  
- Support for QLoRA (`--load_in_4bit`) and [LoRA](./examples/scripts/train_sft_mixtral_lora.sh) (`--lora_rank`, `--target_modules`).  
- Compatibility with HuggingFace's `tokenizer.apply_chat_template` for datasets (`--apply_chat_template` and `--input_key`).  
- Logging support with Wandb (`--use_wandb`) and TensorBoard (`--use_tensorboard`).  
- Checkpoint recovery functionality (`--load_checkpoint` and `--save_steps`).  
- Provided multi-node training scripts, such as [DPO](./examples/scripts/train_llama_slurm.sh) and [Ray PPO](./examples/scripts/train_ppo_llama_ray_slurm.sh).


### PPO Support Matrix

| Feature | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:|
| 70B+ Full Tuning with 16 A100-80GB      | ✅ | ❌ | ❌ | ❌ |
| 7B Full Tuning with 4 RTX4090 | ✅      |    ❌ | ❌ | ❌ |
| 34B DPO Full Tuning with 8 A100-80GB | ✅      |    ❌ | ❌ | ❌ |  
| Inference Engine in PPO | ✅      |    ✅ | ❌ | ❌ |  
| PPO Implementation Tricks | ✅      |    ❌ | ❌ | ✅ |
| Support QLoRA | ✅      |    ❌ | ❌ | ✅ | 
| Support Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ |  
| Support Unmerged Actor-Critic | ✅     |   ✅ | ✅ | ❌ | 
| Support Multiple Reward Models | ✅      |    ❌ | ❌ | ❌ |   
| Support Huggingface Models | ✅      |    ✅ | ✅ | ✅ | 
| Easy-to-use | ✅      |   ❌ (HybridEngine bugs) | ✅ | ✅ | 


## Quick Start

### Installation

To use OpenRLHF, first launch the docker container (**Recommended**) and `pip install` openrlhf inside the docker container:

```bash
# Launch the docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
sudo pip uninstall xgboost transformer_engine flash_attn pynvml -y

# pip install
pip install openrlhf

# If you want to use vLLM acceleration (Install vLLM 0.8.2)
pip install openrlhf[vllm]
# latest vLLM is also supported
pip install openrlhf[vllm_latest]

# pip install the latest version
pip install git+https://github.com/OpenRLHF/OpenRLHF.git

# Or git clone
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!NOTE]
>We recommend using vLLM 0.8.2 or higher.
>`export VLLM_USE_V1=1` requires vLLM 0.8.2 or the Nightly version and enable `export VLLM_ENABLE_V1_MULTIPROCESSING=0`.
>We also provided the [Dockerfiles for vLLM](./dockerfile/) and [One-Click Installation Script of Nvidia-Docker](./examples/scripts/nvidia_docker_install.sh).

### Prepare Datasets
OpenRLHF provides multiple data processing methods in our dataset classes.
Such as in the [Prompt Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6):

```python
def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt
```

- We can use `--input_key` to specify the `JSON key name` of the input datasets `--prompt_data {name or path}` (PPO) or `--dataset {name or path}`, and use `--apply_chat_template` to utilize the `chat_template` from the [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main/en/chat_templating).
- If you don't want to use `--apply_chat_template`, you can use `--input_template` instead, or preprocess the datasets offline in advance.
- OpenRLHF also support mixing multiple datasets using `--prompt_data_probs 0.1,0.4,0.5` (PPO) or `--dataset_probs 0.1,0.4,0.5`.

How Chat Templating Works:

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

How to specify training and test datasets ?

You can specify it using the `data_type@data_dir` format. For example, the dataset can be set as `--dataset json@./data`.

```
data
├── test.jsonl
└── train.jsonl
```

> [!NOTE]
> By default, we use `train` and `test` as splits to distinguish training and testing datasets from Huggingface.
> The ``JSON key`` options depends on the specific datasets. See [Reward Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10) and [SFT Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9)

### Supervised Fine-tuning

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using `--pretrain  {name or path}`, `--reward_pretrain  {name or path}` and `--critic_pretrain  {name or path}`. We have provided some pre-trained checkpoints and datasets on [HuggingFace OpenRLHF](https://huggingface.co/OpenRLHF).

Then you can use the startup scripts we provide in the [examples/scripts](./examples/scripts/) directory, or start the training using the following commands.

```bash 
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --input_template $'User: {}\nAssistant: ' \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# Support HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# Support RingAttention
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# Multi-turn fine-tuning loss
# --multiturn

# Can also be used for continued pre-training
# --pretrain_mode
```

> [!NOTE]
> OpenRLHF SFT/DPO/RewardModel/PPO trainers support `--packing_samples` [based on `--flash_attn`](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)


### Reward Model Training
```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

```

It is recommended to set the `--value_prefix_head` option of the Reward Model to `score`, so that we can load the model using `AutoModelForSequenceClassification`:

```python
reward_model = AutoModelForSequenceClassification.from_pretrained(
              reward_model_path,
              num_labels=1,
              torch_dtype=torch.bfloat16,
              attn_implementation="flash_attention_2",
              use_cache=False,
          )
inputs = xxxx (Left Padding Input Tokens)
reward = reward_model.model(*inputs).last_hidden_state
reward = reward_model.score(reward)[:, -1]
```

### PPO/REINFORCE++ with Ray and vLLM

To improve RLHF training speed or support 70B models, we can use the PPO with Ray and vLLM acceleration (Hybrid Engine)

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep
   --use_wandb {wandb_token}

# Support REINFORCE++  | RLOO | REINFORCE++-baseline | GRPO | Dr. GRPO
# --advantage_estimator reinforce | rloo | reinforce_baseline | group_norm | dr_grpo

# Set --init_kl_coef to 0 will not launch the reference model

# Support remote reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward

# Support N samples
# --n_samples_per_prompt 4
```
> [!NOTE]
> Do not set `--vllm_num_engines` means not using the vLLM engine.
> You can also use ``setup_commands`` to let Ray automatically deploy the environment, such as `--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`.

> [!NOTE]
> RLOO and REINFORCE++-baseline in OPENRLHF are a modification based on REINFORCE++:
> - REINFORCE++ integrates key optimization techniques from PPO (such as advantage normalization and PPO-clip loss) while eliminating the need for a critic network.
> - REINFORCE++-baseline uses the `mean reward of multiple samples from the same prompt` as the baseline to reshape the rewards (with global batch normalization `/std`).
> - RLOO in OpenRLHF modifies the original version by incorporating the `per-token KL reward` and utilizing the `PPO-clip loss`.
> - Dr. GRPO remove the group normalization `/std` in GRPO.


> [!NOTE]
> If you you encounter an error related to index out of range when deepspeed sets up the GPU devices, you can try to set the environment variable [`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`](openrlhf/trainer/ray/utils.py) as a workaround.
>   ```bash
>   # For NVIDIA GPUs:
>   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
>   ```

The launch scripts and documents for supported algorithms are in [example/scripts](./examples/scripts/) and [Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)

## Reinforced Fine-tuning

OpenRLHF supports convenient and efficient Reinforced Fine-tuning. You only need to implement a [file containing the custom `reward_func` function](./examples/scripts/reward_func.py) and pass its path to the `remote_rm_url` parameter. Such as

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    # labels is answers
    print(queries)
    return torch.randn(len(queries))
```

then just set

```shell 
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  ...
  --remote_rm_url /path/to/reward_func.py \
  --label_key answer
```

where the `label_key` parameter is used to pass additional sample information such as answer to the reward function.

### LoRA
If you use `LoRA (Low-Rank Adaptation)`, `OpenRLHF` will not save the full weights by default instead of `LoRA Adapter`. To continue in your task normally, you should combine the `Adapter` with weights of your base model

```bash
python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama3-8b-rm \
    --output_path ./checkpoint/llama-3-8b-rm-combined \
    --is_rm \
    --bf16
```

## Performance

We optimized DSChat's performance to the greatest extent possible by employing techniques such as enabling Adam offload, along with reward model (RM) and reference model (Ref) offload to increase the micro-batch size during the inference stage and avoid out-of-memory issues. We even fixed some bugs in DSChat to enable the Hybrid Engine (HE) for LLaMA2. The average time (seconds) it took to train 1024 prompts with 1 PPO epoch using the Optimized DSChat and OpenRLHF:

| **Size** | **NVIDIA A800-80GB GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

> [!NOTE]
> The data is outdated; please refer to the performance tuning section for re-testing.

### Performance Tuning Guide

To achieve optimal performance, we recommend allocating nodes `vLLM:Actor:Critic = 1:1:1`. 

- For example, for a 70B model with 48 A100 GPUs, it is advised to allocate 16 A100 GPUs to the vLLM Engine, 16 GPUs to the Actor model, and the remaining 16 GPUs to the Critic model. 
- Using hybrid engine `--colocate_all_models` and `--vllm_enable_sleep` and `--deepspeed_enable_sleep` rather than distributed RLHF when there are enough GPU memory.
- Enable the `--colocate_critic_reward`, `--colocate_actor_ref` options to merge nodes.  
- You should increase the `rollout_micro_batch_size` (and minimize the TP size of vLLM engine) as much as possible. During the training phase, a larger `--micro_train_batch_size` is better and enable `--packing_samples`.
- When there are enough GPU memory, please disable `--adam_offload` and enable `--overlap_comm`.
- For vLLM, please use `--vllm_sync_backend nccl` and `export VLLM_USE_V1=1` and `export VLLM_ENABLE_V1_MULTIPROCESSING=0` with vLLM 0.8.2+.   
- Enable [enable_prefix_caching](https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html) in vLLM generation when `n_samples_per_prompts` > 1.
- For a large base model, if an OOM occurs, do not use any `--colocate_xxxx` options.


## Companies and Organizations using OpenRLHF

- Google
- ByteDance
- Tencent
- Alibaba
- Baidu
- China Telecom
- Vivo
- Allen AI
- NexusFlow
- Jülich Supercomputing Centre (JSC)
- Berkeley Starling Team
- M-A-P
- ...

## Join Us

**How to Join?**

1. Email us at janhu9527@gmail.com or join [GitHub Organization](https://github.com/OpenRLHF). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/OpenRLHF).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRLHF/OpenRLHF&type=Date)](https://star-history.com/#OpenRLHF/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRLHF/OpenRLHF" />
</a>

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA ↗](https://llama.meta.com/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design. 
Our project would like to thank [Netmind.AI](https://www.netmind.ai/) for the GPU support of developing ring attention.

(2024/7) Our GitHub organization has changed from OpenLLMAI to OpenRLHF.

## Citation
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```

______________________________________________________________________

*OpenRLHF © 2025 OpenRLHF. All Rights Reserved.*
