import argparse
import torch
import time
import os
import sys
import warnings
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Optional

# --- OpenRLHF Imports (Strategy, Tokenizer) ---
# Add openrlhf parent directory to path if running script directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from openrlhf.utils import get_strategy, get_tokenizer
except ImportError:
    warnings.warn("Could not import get_strategy, get_tokenizer from openrlhf.utils. Ensure OpenRLHF is installed or in PYTHONPATH.", ImportWarning)
    # Define dummy functions if import fails
    def get_strategy(*args, **kwargs): print("Warning: Using dummy get_strategy."); return None
    def get_tokenizer(*args, **kwargs): raise NotImplementedError("Dummy get_tokenizer cannot proceed.")

# --- Local Imports from openrlhf.evaluation --- #

# Harmfulness check function
try:
    from openrlhf.evaluation.harmfulness_metrics import check_bad_token, TARGET_BAD_TOKEN_ID
except ImportError:
    warnings.warn("Could not import from .harmfulness_metrics. Using dummy check_bad.", ImportWarning)
    def check_bad_token(*args, **kwargs) -> bool:
        print("Warning: Using dummy check_bad_token.")
        return False
    TARGET_BAD_TOKEN_ID = -1

# Distribution loading function
try:
    from openrlhf.evaluation.load_lpe_distributions import get_token_factorized_distribution
except ImportError:
    warnings.warn("Could not import from .load_lpe_distributions. Using dummy get_token_factorized_distribution.", ImportWarning)
    def get_token_factorized_distribution(*args, **kwargs) -> list:
        print("Warning: Using dummy get_token_factorized_distribution.")
        # Need to return a list of dummy objects with a 'sample' method for ITGIS/MHIS structure
        class DummyDist:
             def sample(self, shape): return torch.zeros(shape, dtype=torch.long)
             def boltzmann_distribution(self, scores, temperature): return self
        return [DummyDist()]

# --- LPE Method Imports --- #
# Assume low-probability-estimation is installed or importable
try:
    from openrlhf.lpe.methods import ITGIS, MHIS
except ImportError:
    warnings.warn("Could not import ITGIS, MHIS from openrlhf.lpe.methods. LPE will not run.", ImportWarning)
    # Define dummy functions to avoid NameError
    def ITGIS(*args, **kwargs):
        print("Warning: Using dummy ITGIS.")
        return 0.0, 0.0 # Return prob, stderr
    def MHIS(*args, **kwargs):
        print("Warning: Using dummy MHIS.")
        return 0.0, 0.0 # Return prob, stderr

# --- OpenRLHF Model Loading --- #
# This needs to be robust based on how openrlhf saves models
# For now, assume Actor class can load from path
try:
    from openrlhf.models.actor import Actor
except ImportError:
    warnings.warn("Could not import Actor from openrlhf.models.actor. Model loading might fail.", ImportWarning)
    # Use the dummy Actor defined in model_adapters if needed
    try:
        # If adapter import succeeded, it defined a dummy Actor
        pass
    except NameError:
        # Define a basic dummy if adapter also failed
        class Actor(torch.nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, *args, **kwargs): return torch.zeros(1)
            model = torch.nn.Identity()

# --- Add ActorCustom Import ---
try:
    from openrlhf.models.actor_custom import ActorCustom
except ImportError:
    warnings.warn("Could not import ActorCustom from openrlhf.models.actor_custom. Custom parameterization loading will fail.", ImportWarning)
    # Define a dummy if needed, though the logic below will likely prevent its use if Actor is also dummy
    class ActorCustom(Actor): # Inherit dummy Actor if primary failed
        def __init__(self, *args, **kwargs):
             super().__init__(*args, **kwargs)
             print("Warning: Using dummy ActorCustom class.")

def main(args):
    # --- Strategy Setup --- 
    # Set default values for strategy that may not be needed for LPE but are expected by get_strategy
    args.max_steps = 0  # Not training
    args.optim_accumulations = 1 # Not training
    args.bf16 = getattr(args, 'bf16', True) # Ensure bf16 exists
    # Set defaults for args potentially missing if only using minimal strategy setup
    args.adam_offload = getattr(args, 'adam_offload', False)
    args.zpg = getattr(args, 'zpg', 1)
    args.grad_accum_dtype = getattr(args, 'grad_accum_dtype', None)
    args.gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
    args.seed = getattr(args, 'seed', 42) # Add default seed if not present

    strategy = get_strategy(args)
    strategy.setup_distributed()

    # --- Load Model and Tokenizer using Strategy --- #
    # Configure strategy related args
    # Prevent gradient checkpointing during evaluation
    # strategy.args.gradient_checkpointing = False 
    # Already done by Actor init?

    # === Model Instantiation (matching evaluate_gcg_sz.py) ===
    # Create base actor first (using minimal args for DeepSpeed init)
    base_actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.use_flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        # LoRA is applied to the final actor, not necessarily the base
        ds_config=strategy.get_ds_eval_config(offload=False), # Use eval config like GCG base
    )

    # Create the actual actor based on parameterization
    # This mirrors the logic in evaluate_gcg_sz.py
    if "policy" not in args.parameterization:
         strategy.print(f"Using ActorCustom with parameterization: {args.parameterization}")
         actor = ActorCustom(
             args.pretrain,
             initial_model=base_actor, # Pass the base actor
             use_flash_attention_2=args.use_flash_attn,
             bf16=args.bf16,
             load_in_4bit=args.load_in_4bit,
             lora_rank=args.lora_rank,
             lora_alpha=args.lora_alpha,
             target_modules=args.target_modules,
             lora_dropout=args.lora_dropout,
             ds_config=strategy.get_ds_train_config(is_actor=True), # Use train config for potential ZeRO loading
             parameterization=args.parameterization,
             additional_sd_divider=args.additional_sd_divider,
             init_head_from_base=args.init_head_from_base
         )
    else:
        strategy.print("Using standard Actor model (parameterization contains 'policy').")
        # If parameterization is 'policy', reuse the base_actor configuration but potentially add LoRA later if needed?
        # Or should we re-instantiate Actor with LoRA args here? GCG script seems to imply base_actor IS the final actor in this case.
        # Let's follow GCG: the final actor *is* the base_actor instance if parameterization is 'policy'.
        actor = base_actor
        # Apply LoRA here if rank > 0 and parameterization is policy? Actor init doesn't take lora args directly.
        # Need to check how GCG handles LoRA on the base_actor if parameterization=='policy'.
        # GCG's Actor init *does* take LoRA args. Let's re-instantiate Actor if LoRA is needed.
        if args.lora_rank > 0:
             strategy.print(f"Re-instantiating Actor with LoRA config (rank={args.lora_rank})")
             actor = Actor(
                 args.pretrain,
                 use_flash_attention_2=args.use_flash_attn,
                 bf16=args.bf16,
                 load_in_4bit=args.load_in_4bit,
                 lora_rank=args.lora_rank,
                 lora_alpha=args.lora_alpha,
                 target_modules=args.target_modules,
                 lora_dropout=args.lora_dropout,
                 ds_config=strategy.get_ds_train_config(is_actor=True),
             )

    # Prepare the *final* actor model with strategy
    actor = strategy.prepare(actor.eval(), is_rlhf=True)

    # Load checkpoint if specified
    if args.load_checkpoint:
        if not args.ckpt_path:
            strategy.print("Error: --load_checkpoint specified but --ckpt_path is missing.")
            sys.exit(1) # Or raise error
        strategy.print(f"Load checkpoint: {args.ckpt_path}")
        strategy.load_ckpt(actor.model, args.ckpt_path, load_module_strict=True, load_module_only=True)
    else:
        strategy.print("No checkpoint path specified, using base pretrained model.")

    # Get tokenizer
    # Use padding_side='left' like in evaluate_gcg_sz.py for consistency, though LPE might not strictly require it
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy)
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure ID is also set
         strategy.print(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # --- Device Setup (get from strategy) --- 
    # device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    # device = strategy.device # Original line causing error
    device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() and args.local_rank != -1 else torch.device("cuda") # Use local_rank
    strategy.print(f"Using device: {device}")

    # --- Prepare Input Distribution --- #
    strategy.print(f"Preparing input distribution: {args.distribution_name} (length {args.sequence_length})")
    input_dist = get_token_factorized_distribution(
        distribution_name=args.distribution_name, 
        sequence_length=args.sequence_length,
        tokenizer=tokenizer, 
        device=device
    )

    # --- Define the Rare Event Target --- #
    # Convert target word to token ID using the loaded tokenizer
    target_word = args.target_word
    strategy.print(f"Attempting to tokenize target word: '{target_word}'")
    
    # Encode the word, ensuring no special tokens are added that might interfere
    token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    
    # Validate that the word maps to exactly one token ID
    if len(token_ids) == 0:
        raise ValueError(f"Target word '{target_word}' could not be tokenized. Please choose a different word.")
    elif len(token_ids) > 1:
         # Decode the tokens to show the user how it was split
         decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
         strategy.print(
             f"Target word '{target_word}' tokenizes into multiple tokens: {decoded_tokens} (IDs: {token_ids}). "
             f"Proceeding with estimating the probability of this sequence."
         )
    
    # Store the list of target token IDs
    target_token_ids = token_ids # Now this is potentially a list
    # Verify the decoded token matches the input word (optional but good sanity check)
    decoded_check = tokenizer.decode(target_token_ids, skip_special_tokens=True)
    strategy.print(f"Target word '{target_word}' mapped to token IDs: {target_token_ids} (Decodes to: '{decoded_check}')")

    # The LPE methods require the target token ID(s) directly.
    # The check_bad function is for validation/bruteforce.

    # --- Choose and Run Estimator --- # 
    estimator_args = {
        "model": actor, # Pass the prepared Actor model
        "orig_dists": input_dist,
        "target": target_token_ids, # Pass the list of token IDs
        "temp": args.temperature,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "show_progress": args.show_progress,
        "use_argmax": args.use_argmax, # Pass the flag
    }

    print(f"Running {args.method.upper()} estimator...")
    start_time = time.time()
    
    estimated_prob = None
    std_err = None # LPE methods might not return stderr directly

    if args.method.lower() == "itgis":
        # Add ITGIS specific args
        estimator_args["decay_rate"] = args.itgis_decay_rate
        # ITGIS in the provided code returns only the probability estimate
        # We assume stderr calculation would need to be done separately if needed
        estimated_prob = ITGIS(**estimator_args)

    elif args.method.lower() == "mhis":
        # Add MHIS specific args
        estimator_args["burn_in"] = args.mhis_burn_in
        # MHIS in the provided code seems to return only the probability estimate
        estimated_prob = MHIS(**estimator_args)
        
    else:
        raise ValueError(f"Unknown estimation method: {args.method}")

    end_time = time.time()
    print(f"Estimation completed in {end_time - start_time:.2f} seconds.")

    # --- Log Results --- #
    print("\n--- Estimation Results ---")
    print(f"Model Path:       {args.pretrain}")
    print(f"Distribution:     {args.distribution_name} (len={args.sequence_length})")
    print(f"Method:           {args.method.upper()}")
    print(f"Estimation Type:  {'Argmax (Greedy Check)' if args.use_argmax else 'Softmax Probability'}") # Added log
    # Log the target sequence appropriately
    print(f"Target Sequence:  {target_token_ids} ('{target_word}')")
    print(f"Temperature:      {args.temperature}")
    print(f"Num Samples:      {args.n_samples}")
    print(f"Batch Size:       {args.batch_size}")
    if args.method.lower() == "itgis":
        print(f"ITGIS Decay Rate: {args.itgis_decay_rate}")
    if args.method.lower() == "mhis":
        print(f"MHIS Burn-in:     {args.mhis_burn_in}")
        
    print(f"\nEstimated P(target): {estimated_prob:.6e}")

    # --- Save Results (Optional) --- 
    if args.output_file:
        results_data = {
            "model_path": args.pretrain,
            "distribution_name": args.distribution_name,
            "sequence_length": args.sequence_length,
            "method": args.method,
            "target_word": target_word, # Log the original word used
            "target_token_ids": target_token_ids, # Log the actual token IDs
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "use_argmax": args.use_argmax, # Save the flag
            "estimated_probability": estimated_prob,
            "std_err": std_err, # Note: currently N/A for ITGIS/MHIS
            "execution_time_seconds": end_time - start_time,
        }
        if args.method.lower() == "itgis":
            results_data["itgis_decay_rate"] = args.itgis_decay_rate
        if args.method.lower() == "mhis":
            results_data["mhis_burn_in"] = args.mhis_burn_in
            
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save as JSON
        import json
        try:
            with open(args.output_file, 'w') as f:
                json.dump(results_data, f, indent=4)
            print(f"Results saved to: {args.output_file}")
        except Exception as e:
            print(f"Error saving results to {args.output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate rare harmful output probability using LPE methods.")

    # --- Model and Tokenizer Args (aligning with evaluate_gcg_sz.py) --- 
    parser.add_argument("--pretrain", type=str, required=True, help="Path or name of the base pretrained model (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the specific OpenRLHF checkpoint file/dir to load.")
    parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load a checkpoint.")
    # Add args for Actor loading kwargs matching openrlhf.models.actor.Actor init
    parser.add_argument("--use_flash_attn", type=bool, default=False, help="Enable Flash Attention 2 during model loading.")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bfloat16 precision during model loading.")
    parser.add_argument("--load_in_4bit", type=bool, default=False, help="Load model in 4-bit precision.")
    # Add lora args if needed, e.g.:
    # parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank if used during training.")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank for loading adapted models (0 means no LoRA).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--target_modules", type=str, nargs='*', default=None, help="Target modules for LoRA (e.g., 'q_proj' 'v_proj'). Provided as space-separated strings.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout rate.")
    # Add ActorCustom specific args
    parser.add_argument("--parameterization", type=str, default="policy", help="Model parameterization type (e.g., 'policy', 'delta_value').")
    parser.add_argument("--additional_sd_divider", type=float, default=1.0, help="Divider for additional state dict components in ActorCustom.")
    parser.add_argument("--init_head_from_base", action="store_true", help="Initialize ActorCustom head from base model.")

    # Distribution Args
    parser.add_argument("--distribution_name", type=str, required=True, choices=['uniform', 'hex'], help="Name of the input distribution.") # Add more choices as implemented
    parser.add_argument("--sequence_length", type=int, required=True, help="Length of the input sequences for the distribution.")

    # LPE Method Args
    parser.add_argument("--method", type=str, required=True, choices=['itgis', 'mhis'], help="Low-probability estimation method to use.")
    parser.add_argument("--target_word", type=str, required=True, help="The target word to estimate the probability of. Must tokenize to a single token.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature parameter for the LPE method.")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples for the LPE estimator.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LPE estimation.")
    parser.add_argument("--show_progress", action='store_true', help="Show progress bars during estimation.")
    parser.add_argument("--use_argmax", action='store_true', help="Estimate probability based on argmax matching instead of softmax.")

    # ITGIS Specific Args
    parser.add_argument("--itgis_decay_rate", type=float, default=0.9, help="Decay rate for ITGIS gradient EWMA.")
    
    # MHIS Specific Args
    parser.add_argument("--mhis_burn_in", type=int, default=1024, help="Number of burn-in steps for MHIS MCMC.") # Default value needs tuning

    # Output Args
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save results in JSON format.")

    # --- Strategy Args --- 
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO stage (0-3). Use 0 if not using DeepSpeed.")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size for DeepSpeed (needed for strategy init, set to 1 for eval).") # Added for strategy init
    parser.add_argument("--max_len", type=int, default=2048, help="Max sequence length (needed for strategy init).") # Added for strategy init
    # Add other strategy-related args if needed (e.g., adam_offload, zpg) - keeping it minimal for now

    # Other Args
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="Local rank for distributed setups (defaults to -1 for non-distributed).")

    args = parser.parse_args()
    main(args) 