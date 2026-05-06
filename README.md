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

Note: for Setting 4, use a multi-node configuration, such as `bash mk_sb_file.sh --cluster multinode $x`

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
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 0 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0 --start_bonus_alpha 1 --bonus_alpha_schedule linear```

### Setting 2

Baseline command:
```
deepspeed --master_port 35121 --module openrlhf.cli.train_ppo --pretrain distilgpt2 --reward_pretrain nicholasKluge/ToxicityModel --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/exploretoyrlhfmulti03v5 --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/exploretoyrlhfmulti03v5 --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 5 --train_batch_size 5 --micro_rollout_batch_size 1 --rollout_batch_size 1 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 1 --zero_stage 2 --prompt_data Silent-Zebra/this_man_is_a --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 20 --fit_steps 50 --save_info_path /h/319/stephenzhao/OpenRLHF/info/exploretoyrlhfmulti03v5 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-5 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --analytic_calc --new_custom_single_prompt --target_dist_beta -1 --analytic_batch_size 1024 --custom_prompt "2 + 2 =" 
```

For tempering, add:
```--anneal_target_dist_beta --start_target_dist_beta -0.3```

For CFN, add:
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 3 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0```

## Larger Scale Experiments

### Setting 3

First, collect exact target samples. For the train set:
```
deepspeed --master_port 34181 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --apply_chat_template --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestpos --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestpos --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 100 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --fit_steps 1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/itremodevmultitestpos --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --target_dist_beta 20 --reward_cap 5 --rejection_sample_true_target_only --true_target_sample_amount 20 --n_samples_for_f_q 5 --batch_size_rejection_sample 500 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 2000 --rm_max_len 200
```

and for the test set:
```
deepspeed --master_port 30611 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --apply_chat_template --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestposfixedeval --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestposfixedeval --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 100 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --fit_steps 1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/itremodevmultitestposfixedeval --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --target_dist_beta 20 --reward_cap 5 --rejection_sample_true_target_only --true_target_sample_amount 20 --n_samples_for_f_q 5 --batch_size_rejection_sample 500 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 2000 --rm_max_len 200
```

Then run the following (changing file paths to match the output above as needed):

Baseline command:
```
deepspeed --master_port 37971 --module openrlhf.cli.train_ppo --pretrain HuggingFaceTB/SmolLM-135M-Instruct --apply_chat_template --reward_pretrain OpenAssistant/reward-model-deberta-v3-large-v2 --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestposfixedeval0504 --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestposfixedeval0504 --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 250 --train_batch_size 250 --micro_rollout_batch_size 50 --rollout_batch_size 50 --duplicate_rollout_batch_by 5 --max_epochs 1 --prompt_max_len 100 --generate_max_len 20 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --fit_steps 1 --save_info_path /h/319/stephenzhao/OpenRLHF/info/itremodevmultitestposfixedeval0504 --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl_nosecondterm --actor_learning_rate 3e-5 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --target_dist_beta 20 --reward_cap 5 --f_q_g_q_eval --n_samples_for_f_q_g_q 10 --n_prompts_f_q_g_q 50 --load_target_samples_name /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestpos/target_samples_Sm13In_remodev3lav2_rlhf_l20_b20.0_rcap5.0_20misi1_tsa20.pt --f_q_g_q_eval_interval 2000 --rm_max_len 200 --n_eval_prompts_for_f_q 50 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_target_samples_name /h/319/stephenzhao/OpenRLHF/checkpoint/itremodevmultitestposfixedeval/target_samples_Sm13In_remodev3lav2_rlhf_l20_b20.0_rcap5.0_10misi21_tsa20.pt --heldout_input_key prompt
```

For tempering, add:
```--anneal_target_dist_beta --start_target_dist_beta 10```

For CFN, add:
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 0 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0 --start_bonus_alpha 0.3 --bonus_alpha_schedule linear```

### Setting 4

First, collect exact target samples. For the train set:
```
deepspeed --master_port 33361 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/probinfrlhfmulti --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/probinfrlhfmulti --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 20 --micro_rollout_batch_size 5 --rollout_batch_size 5 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --save_info_path /scratch/zhaostep/OpenRLHF/info/probinfrlhfmulti --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --target_dist_beta 50 --reward_cap 8 --rejection_sample_true_target_only --true_target_sample_amount 50 --n_samples_for_f_q_g_q 5 --batch_size_rejection_sample 100 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 1000 --rm_max_len 300
```

and for the test set:
```
deepspeed --master_port 38021 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmultifixedeval --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmultifixedeval --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 20 --micro_rollout_batch_size 5 --rollout_batch_size 5 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/probinfrlhfmultifixedeval --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --target_dist_beta 50 --reward_cap 8 --rejection_sample_true_target_only --true_target_sample_amount 50 --n_samples_for_f_q_g_q 5 --batch_size_rejection_sample 100 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 1000 --rm_max_len 300
```

Then run the following (changing file paths to match the output above as needed):

Baseline command:
```
deepspeed --master_port 35451 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmultifixedeval --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmultifixedeval --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 80 --micro_rollout_batch_size 5 --rollout_batch_size 20 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/20k_mixed_singleturn_1 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/probinfrlhfmultifixedeval --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type rlhf --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-7 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --target_dist_beta 50 --reward_cap 8 --rm_max_len 300 --f_q_g_q_eval --n_samples_for_f_q_g_q 1 --n_prompts_f_q_g_q 50 --load_target_samples_name /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmulti/target_samples_Ll3.1BIn_SkReV2Ll3.1B_rlhf_l100_b50.0_rcap8.0_20misi1_tsa50.pt --f_q_g_q_eval_interval 500 --n_eval_prompts_for_f_q 50 --heldout_prompt_data Silent-Zebra/10k_mixed_singleturn_2_1 --heldout_target_samples_name /h/319/stephenzhao/OpenRLHF/checkpoint/probinfrlhfmultifixedeval/target_samples_Ll3.1BIn_SkReV2Ll3.1B_rlhf_l100_b50.0_rcap8.0_10misi21_tsa50.pt --heldout_input_key prompt 
```

For tempering, add:
```--anneal_target_dist_beta --start_target_dist_beta 20```

For CFN, add:
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 0 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0 --coin_flip_pretrain HuggingFaceTB/SmolLM-135M-Instruct --start_bonus_alpha 0.3 --bonus_alpha_schedule linear```

### Setting 5

First, collect exact target samples. For the train set:
```
deepspeed --master_port 39871 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/indalert --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/indalert --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 20 --micro_rollout_batch_size 5 --rollout_batch_size 5 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/Babelscape_ALERT_selected_short_entries --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/indalert --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type indicator_below_threshold --threshold -5 --target_dist_beta 1 --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --rejection_sample_true_target_only --true_target_sample_amount 40 --n_samples_for_f_q_g_q 5 --batch_size_rejection_sample 100 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 1000 --rm_max_len 300
```

and for the test set:
```
deepspeed --master_port 39521 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /h/319/stephenzhao/OpenRLHF/checkpoint/indalert --ckpt_path /h/319/stephenzhao/OpenRLHF/checkpoint/indalert --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 20 --micro_rollout_batch_size 5 --rollout_batch_size 5 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/ALERT_short_prompts_2 --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 2 --save_info_path /h/319/stephenzhao/OpenRLHF/info/indalert --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type indicator_below_threshold --threshold -5 --target_dist_beta 1 --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 0 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --rejection_sample_true_target_only --true_target_sample_amount 20 --n_samples_for_f_q_g_q 5 --batch_size_rejection_sample 100 --max_gen_per_prompt_rejection 20000 --max_gen_per_prompt_rejection_first_pass 1000 --rm_max_len 300
```

Then run the following (changing file paths to match the output above as needed):

Baseline command (note that I renamed the target samples to ..._tsa36.pt):
```
deepspeed --master_port 38091 --module openrlhf.cli.train_ppo --pretrain meta-llama/Llama-3.2-1B-Instruct --bf16 --reward_pretrain Skywork/Skywork-Reward-V2-Llama-3.2-1B --save_path /scratch/zhaostep/OpenRLHF/checkpoint/indalertt5_100_extra --ckpt_path /scratch/zhaostep/OpenRLHF/checkpoint/indalertt5_100_extra --save_steps -1 --save_steps_harmless -1 --max_ckpt_num 1 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 20 --train_batch_size 20 --micro_rollout_batch_size 5 --rollout_batch_size 5 --duplicate_rollout_batch_by 4 --max_epochs 1 --prompt_max_len 200 --generate_max_len 100 --zero_stage 2 --prompt_data Silent-Zebra/Babelscape_ALERT_selected_short_entries --input_key prompt --apply_chat_template --max_samples 100000 --gradient_checkpointing --num_episodes 1 --do_harmlessness_training --harmlessness_training_num_episodes 100 --save_info_path /scratch/zhaostep/OpenRLHF/info/indalertt5_100_extra --lr_scheduler constant --adam_betas 0.9 0.999 --n_samples_per_prompt 1 --rm_type indicator_below_threshold --threshold -5 --target_dist_beta 1 --seed 1 --parameterization policy_psi_q_p_s_t --actor_loss_type ctl --actor_learning_rate 3e-7 --critic_learning_rate 0 --base_actor_learning_rate 0 --harmlessness_training_loss_type neg_training --reinforce_baseline_type expectation --alpha 0 --init_kl_coef 0 --sampling_iters 1 --f_q_g_q_eval --n_samples_for_f_q_g_q 1 --n_prompts_f_q_g_q 20 --load_target_samples_name /scratch/zhaostep/OpenRLHF/checkpoint/indalert/target_samples_Ll3.1BIn_SkReV2Ll3.1B_indicator_below_threshold_l100_b1.0_thr-5.0_BaALseshen_tsa36.pt --f_q_g_q_eval_interval 50 --n_eval_prompts_for_f_q 20 --heldout_prompt_data Silent-Zebra/ALERT_short_prompts_2 --heldout_target_samples_name /scratch/zhaostep/OpenRLHF/checkpoint/indalert/target_samples_Ll3.1BIn_SkReV2Ll3.1B_indicator_below_threshold_l100_b1.0_thr-5.0_ALshpr2_tsa20.pt --heldout_input_key prompt
```

For tempering, add:
```--start_threshold 5```

For CFN, add:
```--exploration_bonus_sampling_actor coin_flip --bonus_alpha 10 --coin_flip_dim 64 --coin_flip_lr 1e-3 --coin_flip_update_steps 1 --coin_flip_architecture separate_nn --coin_flip_first_online --coin_flip_warmup_steps 0 --coin_flip_pretrain HuggingFaceTB/SmolLM-135M-Instruct```


Don't forget to discuss the exact sample collection code.

Discuss the process of collecting the results also

Then discuss plotting code below


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

