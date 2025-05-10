import argparse
import json
import os
import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Imports from this module
from openrlhf.forecasting_rare_outputs.elicitation import estimate_p_elicit
from openrlhf.forecasting_rare_outputs.forecasting import fit_gumbel_tail, forecast_worst_query_risk
from openrlhf.forecasting_rare_outputs.behavior_defs import get_behavior_definition

# Imports from OpenRLHF
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer, load_model_and_tokenizer, get_strategy
from transformers import PreTrainedModel, PreTrainedTokenizer # Keep for type hints
import torch


def load_queries(query_file: str) -> list[str]:
    """Loads queries from a file (JSONL with 'query' or 'prompt' key, or plain text list)."""
    queries = []
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Query file {query_file} is empty.")
                return []

            # Attempt to parse as JSONL
            is_jsonl = True
            temp_queries_jsonl = []
            for i, line_content in enumerate(lines):
                line_content = line_content.strip()
                if not line_content: continue
                try:
                    data = json.loads(line_content)
                    if 'query' in data and isinstance(data['query'], str):
                        temp_queries_jsonl.append(data['query'])
                    elif 'prompt' in data and isinstance(data['prompt'], str):
                        temp_queries_jsonl.append(data['prompt'])
                    else:
                        # print(f"Warning: Line {i+1} in {query_file} is JSON but lacks 'query'/'prompt' string or has wrong type.")
                        is_jsonl = False; break 
                except json.JSONDecodeError:
                    is_jsonl = False; break # Not JSONL if any line fails
            
            if is_jsonl:
                queries = temp_queries_jsonl
            else:
                # print(f"Could not parse {query_file} as JSONL, attempting to read as plain text lines.")
                queries = [line.strip() for line in lines if line.strip()] # Treat as plain text

        if not queries:
             print(f"Warning: No queries successfully loaded from {query_file} (checked JSONL and plain text)." )
        # else:
             # print(f"Successfully loaded {len(queries)} queries from {query_file}.")
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
                        help="Method for p_elicit estimation ('logprob_specific', 'logprob_keyword', 'repeated_sampling'). Overrides behavior default.")
    parser.add_argument("--k_samples", type=int, default=100,
                        help="Number of samples per query for 'repeated_sampling' or 'logprob_keyword' (if sampling-based).")
    parser.add_argument("--top_k_fit", type=int, default=10,
                        help="Number of top scores (k) to use for Gumbel tail fitting.")
    parser.add_argument("--forecast_scales", "-n", type=str, default="1e4,1e5,1e6,1e7,1e8,1e9",
                        help="Comma-separated list of deployment scales (n) to forecast risk for (e.g., '10000,1000000').")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Maximum number of queries to load from the query file (for debugging/testing).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for query sampling.")

    # --- Generation Parameters (for repeated_sampling / keyword estimation) ---
    parser.add_argument("--gen_temperature", type=float, default=1.0,
                         help="Temperature for sampling generation.")
    parser.add_argument("--gen_max_new_tokens", type=int, default=256,
                         help="Max new tokens for sampling generation.")
    parser.add_argument("--gen_top_p", type=float, default=1.0, # Default to no top-p filtering
                         help="Top-p (nucleus) sampling parameter.")

    args = parser.parse_args()

    # --- Initialize Strategy ---
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # --- Setup & Validation ---
    # Ensure seeding happens after strategy setup if it affects randomness
    strategy.print("Setting random seeds...") # Use strategy print
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
        run_name = f"{model_name_part}_{args.behavior_id}"

    results_file = os.path.join(args.output_dir, f"{run_name}_results.jsonl")
    summary_file = os.path.join(args.output_dir, f"{run_name}_summary.json")

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
    # print(f"Model dtype: {model.model.dtype if hasattr(model, 'model') and hasattr(model.model, 'dtype') else 'N/A'}")
    # Use strategy print for rank 0 logging
    strategy.print(f"Model dtype: {model.module.model.dtype if hasattr(model, 'module') and hasattr(model.module, 'model') and hasattr(model.module.model, 'dtype') else (model.model.dtype if hasattr(model, 'model') and hasattr(model.model, 'dtype') else 'N/A')}")
    # Load checkpoint if provided
    if args.ckpt_path:
        strategy.print(f"Attempting to load checkpoint from: {args.ckpt_path}")
        strategy.load_ckpt(model.model, args.ckpt_path, load_module_strict=True, load_module_only=True)
        strategy.print(f"Checkpoint {args.ckpt_path} loaded successfully.")

    
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
        print(f"Warning: Number of loaded queries ({len(all_queries)}) is less than evaluation_set_size ({m_eval_size}). Using all {len(all_queries)} queries.")
        eval_queries = all_queries
        m_eval_size = len(all_queries)
    else:
        eval_queries = random.sample(all_queries, m_eval_size)
        print(f"Sampled {m_eval_size} queries for evaluation.")

    if m_eval_size == 0:
        print("Error: No queries available for evaluation after sampling/limiting. Exiting.")
        return

    # --- Run Elicitation ---
    print(f"Estimating p_elicit for {m_eval_size} queries using method '{elicitation_method}'...")
    start_time = time.time()
    p_elicits = []
    detailed_results_log = [] 

    generation_params = {}
    if elicitation_method == 'repeated_sampling' or elicitation_method == 'logprob_keyword':
        generation_params = {
            'temperature': args.gen_temperature,
            'max_new_tokens': args.gen_max_new_tokens,
            'top_p': args.gen_top_p,
            'do_sample': True,
        }
        if tokenizer.pad_token_id is not None:
            generation_params['pad_token_id'] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            generation_params['pad_token_id'] = tokenizer.eos_token_id 

    for i, query_text in enumerate(tqdm(eval_queries, desc="Estimating p_elicit")):
        query_log_entry = {"query_idx": i, "query": query_text}
        try:
            p_val = estimate_p_elicit(
                model=model,
                tokenizer=tokenizer,
                query=query_text,
                behavior_definition=behavior_definition,
                method=elicitation_method,
                k_samples=args.k_samples,
                generation_kwargs=generation_params
            )
            p_elicits.append(p_val)
            query_log_entry["p_elicit"] = p_val
        except Exception as e:
            print(f"\nError estimating p_elicit for query_idx {i}: '{query_text[:100]}...'. Error: {e}")
            # import traceback; traceback.print_exc()
            p_elicits.append(np.nan) 
            query_log_entry["p_elicit"] = np.nan
            query_log_entry["error"] = str(e)
        detailed_results_log.append(query_log_entry)

    elicitation_time = time.time() - start_time
    print(f"Elicitation finished in {elicitation_time:.2f} seconds.")

    with open(results_file, 'w', encoding='utf-8') as f_out:
        for record in detailed_results_log:
            f_out.write(json.dumps(record) + '\n')
    print(f"Saved detailed p_elicit results to {results_file}")

    valid_p_elicits = [p for p in p_elicits if p is not None and not np.isnan(p) and np.isfinite(p)]
    num_valid_p = len(valid_p_elicits)
    num_failed_p = m_eval_size - num_valid_p
    print(f"Got {num_valid_p} valid finite p_elicit values ({num_failed_p} failed or non-finite).")

    current_top_k_fit = args.top_k_fit
    if num_valid_p < current_top_k_fit:
        print(f"Warning: Number of valid p_elicit values ({num_valid_p}) is less than top_k_fit ({current_top_k_fit}). Attempting to use all valid points for fitting if >= 2.")
        current_top_k_fit = num_valid_p

    if current_top_k_fit < 2: # Need at least 2 points for linregress
        print(f"Error: Insufficient valid p_elicit values ({current_top_k_fit}) for Gumbel tail fitting (minimum 2 required). Exiting.")
        summary = {
            "run_args": {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()}, # Avoid issues with non-serializable args
            "num_total_queries_loaded": len(all_queries),
            "num_eval_queries_sampled": m_eval_size,
            "num_valid_p_elicits": num_valid_p,
            "num_failed_p_elicits": num_failed_p,
            "elicitation_time_seconds": elicitation_time,
            "error": f"Insufficient valid p_elicit values ({current_top_k_fit}) for fitting."
        }
        with open(summary_file, 'w', encoding='utf-8') as f_summary:
            json.dump(summary, f_summary, indent=4)
        return
    
    print(f"Fitting Gumbel tail using top {current_top_k_fit} scores...")
    a, b, r_value = fit_gumbel_tail(valid_p_elicits, top_k=current_top_k_fit)

    if np.isnan(a) or np.isnan(b):
        print("Error: Gumbel tail fitting failed (resulted in NaN parameters). Cannot proceed with forecasting.")
        error_message = "Gumbel tail fitting failed (NaN params)."
    elif not (np.isfinite(a) and np.isfinite(b)):
        print("Error: Gumbel tail fitting failed (resulted in non-finite parameters). Cannot proceed with forecasting.")
        error_message = "Gumbel tail fitting failed (non-finite params)."
    else:
        error_message = None
        r_squared = r_value**2 if not np.isnan(r_value) else np.nan
        print(f"Fit results: a = {a:.4f}, b = {b:.4f}, r = {r_value:.4f} (R^2 = {r_squared:.4f})")

    forecasted_risks = {}
    if not error_message:
        print(f"Forecasting worst-query risk Q_p(n) for scales n = {forecast_n_values}...")
        for n_scale in forecast_n_values:
            q_p_n_val = forecast_worst_query_risk(a, b, n_scale)
            forecasted_risks[f"Q_p({n_scale})"] = q_p_n_val
            print(f"  Forecast for n={n_scale}: Q_p(n) = {q_p_n_val:.6e}")
    
    summary = {
        "run_args": {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()},
        "pretrain": args.pretrain,
        "behavior_id": args.behavior_id,
        "query_file": args.query_file,
        "elicitation_method": elicitation_method,
        "num_total_queries_loaded": len(all_queries),
        "num_eval_queries_sampled": m_eval_size,
        "num_valid_p_elicits": num_valid_p,
        "num_failed_p_elicits": num_failed_p,
        "elicitation_time_seconds": elicitation_time,
        "gumbel_fit_params": {
            "a_slope": a if not error_message else np.nan,
            "b_intercept": b if not error_message else np.nan,
            "r_value": r_value if not error_message and not np.isnan(r_value) else np.nan,
            "r_squared": r_value**2 if not error_message and not np.isnan(r_value) else np.nan,
            "top_k_actually_used": current_top_k_fit if not error_message else np.nan
        },
        "forecasted_worst_query_risks": forecasted_risks if not error_message else {},
        "error_message": error_message if error_message else None
    }

    with open(summary_file, 'w', encoding='utf-8') as f_summary:
        json.dump(summary, f_summary, indent=4)

    print(f"Experiment finished. Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 