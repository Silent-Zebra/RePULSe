import argparse
import torch
import numpy as np
import os # Added for path manipulation
from openrlhf.utils import get_tokenizer, load_model_and_tokenizer, get_strategy

def main():
    parser = argparse.ArgumentParser(description="Interactive Token Analyzer")

    # --- Model Loading Arguments (mirroring experiment_runner.py) ---
    parser.add_argument("--pretrain", type=str, required=True,
                        help="Path to the Hugging Face model (or identifier).")
    parser.add_argument("--ckpt_path", type=str, default=None, # Added ckpt_path
                        help="Path to the checkpoint to load.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint to load (if applicable).")
    # Add other LoRA args if load_model_and_tokenizer supports them directly or if strategy handles them.
    # For simplicity, focusing on pretrain and lora_path for now.
    # You might need to add lora_rank, lora_alpha, target_modules if your load_model_and_tokenizer uses them.
    
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Whether to use bfloat16 precision.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--flash_attn", action="store_true", default=False,
                        help="Whether to use Flash Attention 2.")

    # --- Deepspeed Arguments (minimal for single GPU interactive use) ---
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training (usually -1 for single GPU).")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO stage (0 typically for inference).")

    args = parser.parse_args()

    # --- Initialize Strategy ---
    # Strategy is used by load_model_and_tokenizer
    strategy = get_strategy(args)
    strategy.setup_distributed() # Minimal setup

    # --- Load Model and Tokenizer ---
    strategy.print("Loading model and tokenizer...")
    # Pass only relevant args to load_model_and_tokenizer
    # Create a new argparse.Namespace or dict for just the args load_model_and_tokenizer expects
    # This is a simplified way; you might need to map args more carefully if load_model_and_tokenizer is strict
    model_load_args = argparse.Namespace(
        pretrain=args.pretrain,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        flash_attn=args.flash_attn,
        # Assuming lora_path is handled within load_model_and_tokenizer or by strategy.load_ckpt later
        # Potentially add: lora_rank=getattr(args, 'lora_rank', 0), lora_alpha=getattr(args, 'lora_alpha', 16), etc.
        # These are not explicitly in your snippet but often part of LoRA loading.
        # For now, assuming load_model_and_tokenizer primarily uses args.pretrain, bf16 etc. and lora_path is separate
    )
    if args.lora_path: # If lora_path is used, it might be loaded differently or might need more args.
        model_load_args.lora_path = args.lora_path
        # You might need to also pass lora_rank, lora_alpha, target_modules here if `load_model_and_tokenizer` uses them with lora_path
        # model_load_args.lora_rank = getattr(args, 'lora_rank', 0) # Example
        # model_load_args.lora_alpha = getattr(args, 'lora_alpha', 16) # Example


    # model, tokenizer = load_model_and_tokenizer(args, strategy) # Original attempt, args might be too broad
    actor_model, tokenizer = load_model_and_tokenizer(model_load_args, strategy)

    strategy.print(f"Base model loaded: {args.pretrain}")

    # --- Load Checkpoint if specified ---
    if args.ckpt_path:
        load_dir = os.path.dirname(args.ckpt_path)
        tag = os.path.basename(args.ckpt_path)
        strategy.print(f"Attempting to load checkpoint from directory: {load_dir} with tag: {tag}")
        
        if not tag: # Handle case where ckpt_path might end with a slash
            strategy.print(f"Warning: Tag derived from ckpt_path '{args.ckpt_path}' is empty. Attempting to use the directory name above.")
            potential_new_tag = os.path.basename(load_dir) # e.g. "my_checkpoint_folder"
            potential_new_load_dir = os.path.dirname(load_dir) # e.g. "/path/to/checkpoints"
            if potential_new_tag:
                strategy.print(f"Revised load_dir: {potential_new_load_dir}, revised tag: {potential_new_tag}")
                load_dir = potential_new_load_dir
                tag = potential_new_tag
            else:
                strategy.print(f"Error: Could not derive a valid tag from {args.ckpt_path}. Checkpoint loading might fail or be incorrect.")
                # Allow to proceed, DeepSpeed might still handle it or error out
        
        # Ensure actor_model.model is the actual nn.Module to load into
        model_to_load_into = actor_model.model if hasattr(actor_model, 'model') else actor_model
        strategy.load_ckpt(model_to_load_into, load_dir, tag=tag, load_module_strict=True, load_module_only=True)
        strategy.print(f"Checkpoint {args.ckpt_path} (tag: {tag}) loaded successfully into the model.")
    elif args.lora_path:
        # This part remains as a note: if lora_path is not handled by load_model_and_tokenizer
        # and represents a PEFT-style LoRA adapter, it might need strategy.load_lora or similar.
        # The current structure assumes load_model_and_tokenizer OR ckpt_path handles adapter loading.
        strategy.print(f"LoRA path specified: {args.lora_path}. Ensure it was applied by load_model_and_tokenizer or consider using ckpt_path for full checkpoints.")

    strategy.print(f"Model loaded: {args.pretrain}")
    if args.lora_path:
        strategy.print(f"With LoRA: {args.lora_path}")
        # Note: The current load_model_and_tokenizer might not directly load LoRA weights with just lora_path.
        # In experiment_runner.py, strategy.load_ckpt is used *after* the initial model load if ckpt_path is provided.
        # If args.lora_path is meant to be a full checkpoint (not just adapters to apply on top of pretrain),
        # the loading logic might need to be adjusted, or ensure load_model_and_tokenizer handles this.
        # For now, assuming `load_model_and_tokenizer` can take `lora_path` and apply it.
        # If LoRA needs to be loaded like a checkpoint, that's a different step:
        # strategy.load_ckpt(actor_model.model, args.lora_path_dir, tag=args.lora_path_tag) # If lora_path was a dir/tag

    actor_model.eval()
    
    try:
        device = actor_model.model.device if hasattr(actor_model, 'model') and hasattr(actor_model.model, 'device') else next(actor_model.parameters()).device
    except StopIteration:
        strategy.print("Warning: Could not determine model device from parameters. Assuming CUDA if available, else CPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    strategy.print(f"Model is on device: {device}")
    strategy.print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab_size: {tokenizer.vocab_size}")
    if tokenizer.bos_token and tokenizer.bos_token_id is not None:
        strategy.print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    else:
        strategy.print("No BOS token configured in tokenizer (or ID is None).")
    if tokenizer.eos_token and tokenizer.eos_token_id is not None:
        strategy.print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    else:
        strategy.print("No EOS token configured in tokenizer (or ID is None).")


    strategy.print("\n--- Interactive Token Analyzer ---")
    strategy.print("Type your text and press Enter. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            input_text = input(">>> ")
            if input_text.lower() in ["quit", "exit"]:
                break
            if not input_text:
                continue

            # Tokenize the input text (without special tokens like BOS/EOS initially)
            token_ids_raw = tokenizer.encode(input_text, add_special_tokens=False)
            
            if not token_ids_raw:
                strategy.print("Input tokenized to an empty sequence.")
                continue

            strategy.print(f"Input text: '{input_text}'")
            strategy.print(f"Token IDs (raw): {token_ids_raw}")
            tokens_decoded = [tokenizer.decode([token_id]) for token_id in token_ids_raw]
            strategy.print(f"Tokens (decoded): {tokens_decoded}")


            # Prepare model input: [BOS] token_1 token_2 ... token_n
            model_input_list = []
            if tokenizer.bos_token_id is not None:
                model_input_list.append(tokenizer.bos_token_id)
            else:
                strategy.print("Warning: BOS token ID is None. Proceeding without prepending BOS.")
            
            model_input_list.extend(token_ids_raw)
            
            if not model_input_list: # Should not happen if token_ids_raw was not empty
                 strategy.print("Cannot proceed with empty model input list.")
                 continue

            model_input_tensor = torch.tensor([model_input_list], dtype=torch.long).to(device)
            
            # Number of "actions" or "target tokens" is the number of tokens from the user's input
            num_actions = len(token_ids_raw)

            if num_actions == 0: # Should be caught by token_ids_raw check
                strategy.print("No tokens from user input to calculate log probabilities for.")
                continue

            # Create a valid attention mask
            attention_mask_tensor = torch.ones_like(model_input_tensor).to(device)

            with torch.no_grad():
                # Get log probabilities P(token_i | BOS, token_0, ..., token_{i-1})
                # The Actor's forward pass with num_actions gives logprobs for the target sequence part.
                action_log_probs = actor_model.forward(
                    model_input_tensor, 
                    num_actions=num_actions, 
                    attention_mask=attention_mask_tensor # Pass the created attention mask
                )
                # actor_model.forward with num_actions is expected to return shape (batch_size, num_actions)
                
                if action_log_probs is None or action_log_probs.ndim == 0:
                    strategy.print("Error: model.forward did not return valid action_log_probs.")
                    continue
                if action_log_probs.shape[0] != 1 or action_log_probs.shape[1] != num_actions:
                    strategy.print(f"Error: action_log_probs shape mismatch. Expected (1, {num_actions}), Got {action_log_probs.shape}")
                    continue

            strategy.print("Log probabilities P(token | preceding_tokens):")
            total_log_prob = 0.0
            for i in range(num_actions):
                token_id = token_ids_raw[i]
                token_str = tokenizer.decode([token_id])
                log_prob = action_log_probs[0, i].item()
                total_log_prob += log_prob
                strategy.print(f"  Token {i+1:2d}: '{token_str}' (ID: {token_id:5d}) 	 LogP: {log_prob:.4f}")
            
            avg_log_prob = total_log_prob / num_actions if num_actions > 0 else 0
            perplexity = np.exp(-avg_log_prob) if num_actions > 0 and total_log_prob != 0 else float('inf') # Or handle differently

            strategy.print(f"Total LogP of sequence: {total_log_prob:.4f}")
            strategy.print(f"Average LogP per token: {avg_log_prob:.4f}")
            strategy.print(f"Perplexity of sequence: {perplexity:.4f}")


        except EOFError: # Handle Ctrl+D
            strategy.print("\nExiting...")
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            strategy.print("\nExiting...")
            break
        except Exception as e:
            strategy.print(f"An error occurred on line {e.__traceback__.tb_lineno}: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main() 