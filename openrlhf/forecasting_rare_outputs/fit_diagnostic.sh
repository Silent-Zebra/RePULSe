BEHAVIOR_ID=violence_glorification
TOP_K_FIT=4

newenv/bin/python -m openrlhf.forecasting_rare_outputs.fit_diagnostic \
  --input_file "openrlhf/forecasting_rare_outputs/results/nobootstrap_top4_fit/13-05-2025/main-q-prop/{seed}/${BEHAVIOR_ID}/${BEHAVIOR_ID}-logprob_target_seq-visualization-data.json" \
  --seeds s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 \
  --output_dir "openrlhf/forecasting_rare_outputs/diagnostic_plots/nobootstrap_top${TOP_K_FIT}_fit" \
  --base_filename "multi-seed-k${TOP_K_FIT}-${BEHAVIOR_ID}_fit" \
  --k_fit "${TOP_K_FIT}"