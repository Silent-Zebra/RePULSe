TOP_K=4

./newenv/bin/python openrlhf/forecasting_rare_outputs/analysis.py \
  --base_dir openrlhf/forecasting_rare_outputs/results/nobootstrap_top4_fit/13-05-2025/ \
  --output_dir openrlhf/forecasting_rare_outputs/analysis_results_nobootstrap_top${TOP_K}_fit/ \
  --k_fit $TOP_K \
  --behavior_agg "mean" \
  --seed_agg "mean"
#  --ci

