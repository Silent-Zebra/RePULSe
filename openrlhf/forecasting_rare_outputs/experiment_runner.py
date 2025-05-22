import argparse
import json
import os
import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from openrlhf.forecasting_rare_outputs.elicitation import estimate_p_elicit
from openrlhf.forecasting_rare_outputs.forecasting import fit_gumbel_tail, forecast_worst_query_risk
from openrlhf.forecasting_rare_outputs.behavior_defs import get_behavior_definition

from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer, load_model_and_tokenizer, get_strategy
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch


def load_queries(query_file: str) -> list[str]:
    """Loads queries from a file (JSONL with 'query', 'prompt', or nested 'prompt[0].content' key, or plain text list)."""
    queries = []
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Query file {query_file} is empty.")
                return []

            potential_queries = []
            all_lines_json_parsable_with_key = True

            for i, line_content in enumerate(lines):
                line_content = line_content.strip()
                if not line_content:
                    continue

                extracted_query = None
                try:
                    data = json.loads(line_content)
                    if isinstance(data, dict):
                        if 'query' in data and isinstance(data['query'], str):
                            extracted_query = data['query']
                        elif 'prompt' in data:
                            if isinstance(data['prompt'], str):
                                extracted_query = data['prompt'] # Handles {"prompt": "query text"}
                            elif isinstance(data['prompt'], list) and data['prompt']:
                                first_prompt_item = data['prompt'][0]
                                if isinstance(first_prompt_item, dict) and 'content' in first_prompt_item and isinstance(first_prompt_item['content'], str):
                                    extracted_query = first_prompt_item['content'] # Handles {"prompt": [{"content": "query text", ...}]}
                        
                        if extracted_query is not None:
                            potential_queries.append(extracted_query)
                        else:
                            # Parsed as JSON, but didn't find the expected keys/structure
                            all_lines_json_parsable_with_key = False
                            break 
                    else:
                        # Parsed as JSON, but it's not a dictionary (e.g., a JSON list, number)
                        all_lines_json_parsable_with_key = False
                        break
                except json.JSONDecodeError:
                    # Not JSON, or malformed JSON
                    all_lines_json_parsable_with_key = False
                    break 
            
            if all_lines_json_parsable_with_key:
                queries = potential_queries
            else:
                # Fallback: Treat as plain text if any line wasn't JSON or didn't match structure
                print(f"Could not parse {query_file} consistently as JSONL with expected keys, attempting to read as plain text lines.")
                queries = [line.strip() for line in lines if line.strip()]

        if not queries:
            print(f"Warning: No queries successfully loaded from {query_file} (checked JSONL types and plain text).")
        else:
            print(f"Successfully loaded {len(queries)} queries from {query_file}.")
        return queries

    except FileNotFoundError:
        print(f"Error: Query file not found at {query_file}")
        raise
    except Exception as e:
        print(f"Error loading or processing queries from {query_file}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run Jones et al. (2025) EVT Forecasting Experiment")

    # --- Core Arguments ---
    parser.add_argument("--pretrain", type=str, required=True,
                        help="Path to the Hugging Face model (or identifier).")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--behavior_id", type=str, required=True,
                        help="ID of the behavior definition from behavior_defs.py (e.g., 'chem_misuse_chlorine_specific')")
    parser.add_argument("--query_file", type=str, required=True,
                        help="Path to the file containing evaluation queries (e.g., JSONL file with 'query' field).")
    parser.add_argument("--output_dir", type=str, default="./openrlhf/forecasting_rare_outputs/results",
                        help="Directory to save the results.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional name for this run (used for output filename). Defaults to model_name + behavior_id.")

    # --- Model Loading Arguments (align with Actor constructor where possible) ---
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint to load (if applicable).")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="Rank of LoRA adapters (used if lora_path is specified).")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha value for LoRA scaling.")
    parser.add_argument("--target_modules", type=str, nargs='*' , default=None,
                        help="Which modules to apply LoRA to (e.g. ['q_proj', 'v_proj'])")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Whether to use bfloat16 precision.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--flash_attn", action="store_true", default=False,
                        help="Whether to use Flash Attention 2.")

    # --- Deepspeed Arguments ---
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO stage (0-3)")

    # --- Evaluation & Forecasting Parameters ---
    parser.add_argument("--evaluation_set_size", "-m", type=int, default=1000,
                        help="Number of queries (m) to sample from the pool for evaluation.")
    parser.add_argument("--elicitation_method", type=str, default=None,
                        help="Method for p_elicit estimation ('logprob_target_seq', 'logprob_target_keyword_in_target_seq', 'repeated_sampling'). Overrides behavior default.")
    parser.add_argument("--k_samples", type=int, default=100,
                        help="Number of samples per query for 'repeated_sampling'.")
    parser.add_argument("--top_k_fit", type=int, default=10,
                        help="Number of top scores (k) to use for Gumbel tail fitting.")
    parser.add_argument("--forecast_scales", "-n", type=str, default="1e4,1e5,1e6,1e7,1e8,1e9",
                        help="Comma-separated list of deployment scales (n) to forecast risk for (e.g., '10000,1000000').")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Maximum number of queries to load from the query file (for debugging/testing).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for query sampling.")

    # --- Generation Parameters (primarily for repeated_sampling) ---
    parser.add_argument("--gen_temperature", type=float, default=1.0,
                         help="Temperature for sampling generation (used by 'repeated_sampling').")
    parser.add_argument("--gen_max_new_tokens", type=int, default=256,
                         help="Max new tokens for sampling generation (used by 'repeated_sampling').")
    parser.add_argument("--gen_top_p", type=float, default=1.0, # Default to no top-p filtering
                         help="Top-p (nucleus) sampling parameter (used by 'repeated_sampling').")
    parser.add_argument("--gen_top_k", type=int, default=None, 
                         help="Top-k sampling parameter (used by 'repeated_sampling').")
    parser.add_argument("--elicitation_processing_batch_size", type=int, default=1000,
                        help="For 'repeated_sampling', the number of samples to generate and process in each internal batch within estimate_p_elicit. Default is 1000.")

    # --- New parameters for repeated_sampling
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for p_elicit estimation and generation.")
    parser.add_argument("--num_bootstrap_samples", type=int, default=1,
                        help="Number of bootstrap samples to draw for Gumbel parameter estimation. Default is 1 (no bootstrapping).")

    args = parser.parse_args()

    # --- Initialize Strategy ---
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # --- Setup & Validation ---
    # Ensure seeding happens after strategy setup if it affects randomness
    strategy.print("Setting random seeds...")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    forecast_n_values = [int(float(n_str)) for n_str in args.forecast_scales.split(',') if n_str]
    if not forecast_n_values:
        raise ValueError("No valid forecast scales (n) provided.")

    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.run_name
    if run_name is None:
        model_name_part = os.path.basename(args.pretrain.rstrip('/'))
        run_name = f"{args.behavior_id}-{args.elicitation_method}"

    results_file = os.path.join(args.output_dir, f"{run_name}-results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{run_name}-summary.json")

    print(f"Starting experiment run: {run_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Arguments: {vars(args)}")

    # --- Load Resources ---
    print("Loading behavior definition...")
    behavior_definition = get_behavior_definition(args.behavior_id)
    print(f"Loaded behavior: {behavior_definition.id}, {behavior_definition.description}")

    elicitation_method = args.elicitation_method or behavior_definition.default_elicitation_method
    if not elicitation_method:
        raise ValueError(f"Elicitation method must be specified for '{args.behavior_id}' or its behavior definition")
    print(f"Using elicitation method: {elicitation_method}")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args, strategy)
    print(f"Model dtype: {model.module.model.dtype if hasattr(model, 'module') and hasattr(model.module, 'model') and hasattr(model.module.model, 'dtype') else (model.model.dtype if hasattr(model, 'model') and hasattr(model.model, 'dtype') else 'N/A')}")
    strategy.print(f"Model dtype: {model.module.model.dtype if hasattr(model, 'module') and hasattr(model.module, 'model') and hasattr(model.module.model, 'dtype') else (model.model.dtype if hasattr(model, 'model') and hasattr(model.model, 'dtype') else 'N/A')}")

    if args.ckpt_path:
        load_dir = os.path.dirname(args.ckpt_path)
        tag = os.path.basename(args.ckpt_path)
        strategy.print(f"Attempting to load checkpoint from directory: {load_dir} with tag: {tag}")
        if not tag:
            strategy.print(f"Warning: Tag derived from ckpt_path '{args.ckpt_path}' is empty. Attempting to use the directory name above.")
            potential_new_tag = os.path.basename(load_dir)
            potential_new_load_dir = os.path.dirname(load_dir)
            if potential_new_tag:
                 strategy.print(f"Revised load_dir: {potential_new_load_dir}, revised tag: {potential_new_tag}")
                 load_dir = potential_new_load_dir
                 tag = potential_new_tag
            else:
                strategy.print(f"Error: Could not derive a valid tag from {args.ckpt_path}. Checkpoint loading might fail.")
                
        strategy.load_ckpt(model.model, load_dir, tag=tag, load_module_strict=True, load_module_only=True)
        strategy.print(f"Checkpoint {args.ckpt_path} (tag: {tag}) loaded successfully.")

    print(f"Loading queries from {args.query_file}...")
    all_queries = load_queries(args.query_file)
    if args.max_queries is not None and args.max_queries > 0:
        all_queries = all_queries[:args.max_queries]
    print(f"Loaded {len(all_queries)} total queries for run.")

    if len(all_queries) == 0:
        print("Error: No queries loaded. Exiting.")
        return

    m_eval_size = args.evaluation_set_size
    if len(all_queries) < m_eval_size:
        print(f"Warning: Number of loaded queries ({len(all_queries)}) is less than evaluation_set_size ({m_eval_size}). Using all {len(all_queries)} queries for each bootstrap sample (if bootstrapping).")
        
    # Initialize lists to store parameters from each bootstrap iteration
    bootstrap_a_params = []
    bootstrap_b_params = []
    bootstrap_r_values = []
    bootstrap_num_valid_p_elicits = []
    bootstrap_top_k_actually_used = []
    total_elicitation_time = 0
    detailed_results_log_iter = [] # Initialize for the first successful bootstrap iteration's log

    # --- Bootstrap Loop ---
    num_bootstrap_iterations = args.num_bootstrap_samples
    print(f"Starting {num_bootstrap_iterations} bootstrap iterations...")

    for bootstrap_iter in range(num_bootstrap_iterations):
        strategy.print(f"--- Bootstrap Iteration {bootstrap_iter + 1} / {num_bootstrap_iterations} ---")

        a_iter, b_iter, r_value_iter = np.nan, np.nan, np.nan 
                                       
        while True:
            current_m_eval_size = m_eval_size

            if len(all_queries) == 0:
                strategy.print(f"CRITICAL Error: all_queries is empty at iter {bootstrap_iter + 1}. Cannot proceed. Check query loading and initial checks.")
                raise ValueError("all_queries is empty at iter {bootstrap_iter + 1}. Cannot proceed. Check query loading and initial checks.")

            elif len(all_queries) < current_m_eval_size:
                strategy.print(f"  Requested m_eval_size ({current_m_eval_size}) is > total available unique queries ({len(all_queries)}). Sampling {current_m_eval_size} queries with replacement from the {len(all_queries)} available queries for iter {bootstrap_iter + 1} attempt.")

            eval_queries = random.choices(all_queries, k=current_m_eval_size)
            
            if not eval_queries:
                strategy.print(f"Warning: No queries sampled for evaluation in iter {bootstrap_iter + 1} attempt (m_eval_size: {current_m_eval_size}, all_queries_len: {len(all_queries)}). Retrying sampling...")
                time.sleep(0.1)
                continue

            # --- Run Elicitation for the current bootstrap sample attempt ---
            strategy.print(f"  Estimating p_elicit for {len(eval_queries)} queries using method '{elicitation_method}' (Iter {bootstrap_iter + 1} attempt)...")
            attempt_start_time = time.time() # Time for this attempt's elicitation
            p_elicits_iter_attempt = []
            
            # For storing detailed logs of the first bootstrap iteration's successful attempt
            current_attempt_detailed_results_log = [] 

            current_generation_kwargs = {}
            if elicitation_method == 'repeated_sampling':
                current_generation_kwargs = {
                    'temperature': args.gen_temperature,
                    'max_new_tokens': args.gen_max_new_tokens,
                    'top_p': args.gen_top_p,
                    'top_k': args.gen_top_k,
                    'do_sample': True,
                }
                if tokenizer.pad_token_id is not None:
                    current_generation_kwargs['pad_token_id'] = tokenizer.pad_token_id
                elif tokenizer.eos_token_id is not None:
                    current_generation_kwargs['pad_token_id'] = tokenizer.eos_token_id

            for i, query_text in enumerate(tqdm(eval_queries, desc=f"Estimating p_elicit (Iter {bootstrap_iter+1} attempt)", disable=False)):
                query_log_entry = {"query_idx_bootstrap": i, "query": query_text, "bootstrap_iteration": bootstrap_iter + 1, "attempt_in_iter": "current"} # Placeholder for attempt count if needed
                try:
                    p_val = estimate_p_elicit(
                        model=model,
                        tokenizer=tokenizer,
                        query=query_text,
                        behavior_definition=behavior_definition,
                        method=elicitation_method,
                        k_samples=args.k_samples if elicitation_method == 'repeated_sampling' else 0,
                        generation_kwargs=current_generation_kwargs if elicitation_method == 'repeated_sampling' else {},
                        elicitation_processing_batch_size=args.elicitation_processing_batch_size if elicitation_method == 'repeated_sampling' else 1000
                    )
                    p_elicits_iter_attempt.append(p_val)
                    query_log_entry["p_elicit"] = p_val
                except Exception as e:
                    strategy.print(f"\nError estimating p_elicit for query_idx {i} in bootstrap iter {bootstrap_iter+1} attempt: '{query_text[:100]}...'. Error: {e}")
                    p_elicits_iter_attempt.append(np.nan) # Append NaN for this query
                    query_log_entry["p_elicit"] = np.nan
                    query_log_entry["error"] = str(e)
                
                if bootstrap_iter == 0: # Only collect detailed log entries for the first bootstrap iteration
                    current_attempt_detailed_results_log.append(query_log_entry)

            attempt_elicitation_time = time.time() - attempt_start_time
            strategy.print(f"  Elicitation for iter {bootstrap_iter + 1} attempt finished in {attempt_elicitation_time:.2f} seconds.")

            valid_p_elicits_iter = [p for p in p_elicits_iter_attempt if p is not None and not np.isnan(p) and np.isfinite(p)]
            num_valid_p_iter = len(valid_p_elicits_iter)
            num_failed_p_iter = len(eval_queries) - num_valid_p_iter
            
            strategy.print(f"  Got {num_valid_p_iter} valid finite p_elicit values ({num_failed_p_iter} failed or non-finite) for iter {bootstrap_iter + 1} attempt.")

            current_top_k_fit_iter = args.top_k_fit
            if num_valid_p_iter < current_top_k_fit_iter:
                strategy.print(f"  Warning: Number of valid p_elicit values ({num_valid_p_iter}) is less than top_k_fit ({current_top_k_fit_iter}) for iter {bootstrap_iter + 1} attempt. Using all {num_valid_p_iter} valid points for fitting if >= 2.")
                current_top_k_fit_iter = num_valid_p_iter
            
            if current_top_k_fit_iter < 2:
                strategy.print(f"  Error: Insufficient valid p_elicit values ({current_top_k_fit_iter}) for Gumbel tail fitting (minimum 2 required) for iter {bootstrap_iter + 1} attempt. Retrying this bootstrap sample...")
                time.sleep(0.1) 
                continue # Retry the while loop: re-sample, re-elicit

            strategy.print(f"  Fitting Gumbel tail using top {current_top_k_fit_iter} scores for iter {bootstrap_iter + 1} attempt...")
            # These are specific to this attempt
            a_iter_attempt, b_iter_attempt, r_value_iter_attempt = fit_gumbel_tail(valid_p_elicits_iter, top_k=current_top_k_fit_iter)

            if np.isnan(a_iter_attempt) or np.isnan(b_iter_attempt) or not (np.isfinite(a_iter_attempt) and np.isfinite(b_iter_attempt)):
                strategy.print(f"  Warning: Gumbel tail fitting for iter {bootstrap_iter + 1} attempt resulted in NaN/non-finite parameters (a={a_iter_attempt}, b={b_iter_attempt}). Retrying this bootstrap sample...")
                time.sleep(0.1)
                continue # Retry the while loop: re-sample, re-elicit
            elif abs(r_value_iter_attempt) < 0.8:
                r_squared_iter_attempt = r_value_iter_attempt**2 if not np.isnan(r_value_iter_attempt) else np.nan
                strategy.print(f"  Warning: Fit for Bootstrap Iteration {bootstrap_iter + 1} attempt has R-value ({r_value_iter_attempt:.4f}, abs={abs(r_value_iter_attempt):.4f}, R^2={r_squared_iter_attempt:.4f}) below 0.8. Discarding this Gumbel fit. This iteration will not contribute parameters.")
                # This attempt's Gumbel parameters are not stored.
                # total_elicitation_time is not incremented for this discarded fit.
                # detailed_results_log_iter is not updated with this attempt's log.
                break # Exit the `while True` loop for this `bootstrap_iter`. Outer loop proceeds to next `bootstrap_iter`.
            else:
                # Successful fit for this attempt: parameters are valid AND abs(r_value) >= 0.8.
                a_iter, b_iter, r_value_iter = a_iter_attempt, b_iter_attempt, r_value_iter_attempt # Assign to outer loop scope vars
                
                r_squared_iter = r_value_iter**2 if not np.isnan(r_value_iter) else np.nan
                strategy.print(f"  SUCCESSFUL Fit for Bootstrap Iteration {bootstrap_iter + 1}: a = {a_iter:.4f}, b = {b_iter:.4f}, r = {r_value_iter:.4f} (R^2 = {r_squared_iter:.4f})")
                
                bootstrap_a_params.append(a_iter)
                bootstrap_b_params.append(b_iter)
                bootstrap_r_values.append(r_value_iter)
                bootstrap_num_valid_p_elicits.append(num_valid_p_iter)
                bootstrap_top_k_actually_used.append(current_top_k_fit_iter)
                
                total_elicitation_time += attempt_elicitation_time

                if len(bootstrap_a_params) == 1: 
                    detailed_results_log_iter = current_attempt_detailed_results_log
                
                break

    # --- End of Bootstrap Loop ---

    # Save detailed p_elicit results (only from the first successful bootstrap iteration)
    if detailed_results_log_iter: 
        with open(results_file, 'w', encoding='utf-8') as f_out:
            for record in detailed_results_log_iter:
                f_out.write(json.dumps(record) + '\n')
        print(f"Saved detailed p_elicit results (from first bootstrap iteration) to {results_file}")
    else:
        print(f"No detailed p_elicit results to save (e.g. all bootstrap iterations may have failed early or num_bootstrap_samples was 0).")

    # --- Process Bootstrap Results ---
    # Calculate mean and std dev for Gumbel parameters, handling NaNs
    mean_a = np.nanmean(bootstrap_a_params) if bootstrap_a_params else np.nan
    std_a = np.nanstd(bootstrap_a_params) if bootstrap_a_params else np.nan
    mean_b = np.nanmean(bootstrap_b_params) if bootstrap_b_params else np.nan
    std_b = np.nanstd(bootstrap_b_params) if bootstrap_b_params else np.nan
    mean_r_value = np.nanmean(bootstrap_r_values) if bootstrap_r_values else np.nan
    std_r_value = np.nanstd(bootstrap_r_values) if bootstrap_r_values else np.nan
    
    mean_num_valid_p = np.nanmean(bootstrap_num_valid_p_elicits) if bootstrap_num_valid_p_elicits else np.nan
    mean_top_k_used = np.nanmean(bootstrap_top_k_actually_used) if bootstrap_top_k_actually_used else np.nan
    
    num_successful_fits = sum(1 for a, b in zip(bootstrap_a_params, bootstrap_b_params) if not (np.isnan(a) or np.isnan(b)))

    print(f"\n--- Bootstrap Aggregation Results ({num_bootstrap_iterations} iterations) ---")
    print(f"Number of successful Gumbel fits: {num_successful_fits} / {num_bootstrap_iterations}")
    if num_successful_fits > 0:
        print(f"Mean a: {mean_a:.4f} (Std: {std_a:.4f})")
        print(f"Mean b: {mean_b:.4f} (Std: {std_b:.4f})")
        mean_r_squared = mean_r_value**2 if not np.isnan(mean_r_value) else np.nan # Note: mean(r^2) is not (mean(r))^2
                                                                                   # For simplicity, reporting based on mean_r_value
        print(f"Mean r_value: {mean_r_value:.4f} (Std: {std_r_value:.4f}), Implied Mean R^2: {mean_r_squared:.4f}")
        print(f"Mean number of valid p_elicits per iteration: {mean_num_valid_p:.2f}")
        print(f"Mean top_k actually used for fitting per iteration: {mean_top_k_used:.2f}")
    else:
        print("No successful Gumbel fits across all bootstrap iterations.")

    overall_error_message = None
    if num_successful_fits == 0 and num_bootstrap_iterations > 0:
        overall_error_message = f"All {num_bootstrap_iterations} bootstrap Gumbel fits failed."
    elif np.isnan(mean_a) or np.isnan(mean_b):
        overall_error_message = "Mean Gumbel parameters are NaN (likely due to too many fit failures)."
    elif not (np.isfinite(mean_a) and np.isfinite(mean_b)):
        overall_error_message = "Mean Gumbel parameters are non-finite."
    
    # Use mean parameters for forecasting
    final_a_for_forecast = mean_a
    final_b_for_forecast = mean_b

    forecasted_risks = {}
    if not overall_error_message:
        print(f"\nForecasting worst-query risk Q_p(n) using mean Gumbel parameters (a={final_a_for_forecast:.4f}, b={final_b_for_forecast:.4f}) for scales n = {forecast_n_values}...")
        for n_scale in forecast_n_values:
            q_p_n_val = forecast_worst_query_risk(final_a_for_forecast, final_b_for_forecast, n_scale)
            forecasted_risks[f"Q_p({n_scale})"] = q_p_n_val
            print(f"  Forecast for n={n_scale}: Q_p(n) = {q_p_n_val:.6e}")
    else:
        print(f"Error: Cannot proceed with forecasting due to: {overall_error_message}")

    # --- Prepare Summary Output ---
    # Calculate average m_eval_size used, especially if it was adjusted
    # For bootstrap, m_eval_size is fixed per iteration by args.evaluation_set_size or len(all_queries) if smaller
    # The critical part is the number of queries from which bootstrap samples were drawn.
    
    actual_m_eval_size_used_per_bootstrap = args.evaluation_set_size # This is the k for random.choices

    summary = {
        "run_args": {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()},
        "pretrain": args.pretrain,
        "behavior_id": args.behavior_id,
        "query_file": args.query_file,
        "elicitation_method": elicitation_method,
        "num_total_queries_loaded": len(all_queries),
        "num_bootstrap_samples_requested": num_bootstrap_iterations,
        "num_bootstrap_samples_completed": len(bootstrap_a_params), # Iterations that at least started fitting
        "num_successful_gumbel_fits": num_successful_fits,
        "evaluation_set_size_m_per_bootstrap": actual_m_eval_size_used_per_bootstrap,
        "mean_num_valid_p_elicits_per_bootstrap": mean_num_valid_p if not np.isnan(mean_num_valid_p) else None,
        "total_elicitation_time_seconds": total_elicitation_time,
        "gumbel_fit_params_bootstrap_summary": {
            "mean_a_slope": final_a_for_forecast if not overall_error_message else np.nan,
            "std_a_slope": std_a if not overall_error_message and not np.isnan(std_a) else np.nan,
            "mean_b_intercept": final_b_for_forecast if not overall_error_message else np.nan,
            "std_b_intercept": std_b if not overall_error_message and not np.isnan(std_b) else np.nan,
            "mean_r_value": mean_r_value if not overall_error_message and not np.isnan(mean_r_value) else np.nan,
            "std_r_value": std_r_value if not overall_error_message and not np.isnan(std_r_value) else np.nan,
            "mean_r_squared_approx": (mean_r_value**2) if not overall_error_message and not np.isnan(mean_r_value) else np.nan,
            "mean_top_k_actually_used": mean_top_k_used if not overall_error_message and not np.isnan(mean_top_k_used) else np.nan
        },
        "bootstrap_iterations_data": {
            "a_slopes": [p if not np.isnan(p) else None for p in bootstrap_a_params], # Replace NaN with None for JSON
            "b_intercepts": [p if not np.isnan(p) else None for p in bootstrap_b_params],
            "r_values": [p if not np.isnan(p) else None for p in bootstrap_r_values],
            "num_valid_p_elicits": bootstrap_num_valid_p_elicits,
            "top_k_fits": bootstrap_top_k_actually_used
        },
        "forecasted_worst_query_risks_from_mean_params": forecasted_risks if not overall_error_message else {},
        "overall_error_message": overall_error_message
    }

    with open(summary_file, 'w', encoding='utf-8') as f_summary:
        json.dump(summary, f_summary, indent=4)

    print(f"Experiment finished. Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 