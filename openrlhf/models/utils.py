from typing import Optional, Tuple, Union

import bitsandbytes as bnb
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    # print("--ACTION MASK--")
    # print(action_mask.size())
    # print(action_mask)
    # print("--KL LOG RATIO--")
    # print(log_ratio.mean())
    # print(log_ratio)
    # print(log_ratio.size())
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # print("--KL LOG RATIO AFTER MASK--")
    # print(log_ratio.mean())
    # print(log_ratio)

    if log_ratio.mean() < 0: # Diagnostic
        print("--LOG PROBS--")
        print(log_probs.mean())
        print(log_probs)
        # print("--LOG PROBS AFTER MASK--")
        # print((log_probs * action_mask).mean())
        # print(log_probs * action_mask)
        print("--LOG PROBS BASE--")
        print(log_probs_base.mean())
        print(log_probs_base)
        # print("--LOG PROBS BASE AFTER MASK--")
        # print((log_probs_base * action_mask).mean())
        # print(log_probs_base * action_mask)
        # for i in range(log_probs.shape[0]):
        for i in range(1):
            print(f"---{i}--")
            print(log_probs[i])
            print(log_probs_base[i])
            # print(action_mask[i])




    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    clamp_reward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    print("action_mask")
    print(action_mask)

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -kl_coef * kl
    # print("--KL REWARD--")
    # print(kl_reward.mean())
    # print(kl_reward)


    # print("REWARD BEFORE CLAMP")
    # print(r)
    if clamp_reward:
        r = r.clamp(min=-10, max=10)

    # print("REWARD AFTER CLAMP")
    # print(r)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    print("--EOS INDICES--")
    print(eos_indices.squeeze(-1))

    print("--LAST REWARD--")
    # print(last_reward.mean())
    print(last_reward.shape)
    print(last_reward.sum(-1))
    print(last_reward)

    print("--KL REWARD--")
    print(kl_reward.shape)
    print(kl_reward.sum(-1))
    print(kl_reward)

    reward = last_reward + kl_reward
    return reward, kl


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, return_type: str = 'p', return_unnormalized=False) -> torch.Tensor:
    if return_unnormalized:
        return return_or_gather_then_return(labels, logits, return_type)

    log_probs_all_vocab = F.log_softmax(logits, dim=-1)
    # print("log probs full")
    # print(log_probs)
    # print("log probs max")
    # print(log_probs.max(dim=-1))
    return return_or_gather_then_return(labels, log_probs_all_vocab, return_type)


def return_or_gather_then_return(labels, log_probs_all_vocab, return_type):
    if return_type == "all_vocab":
        return log_probs_all_vocab
    assert labels is not None
    # Select logits for the particular next tokens that were generated (are in the sequence/are the 'labels')
    log_probs_labels = log_probs_all_vocab.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_type == "both":  # produces all_vocab, then just for the specific labels
        return log_probs_all_vocab, log_probs_labels
    return log_probs_labels


def log_probs_from_logits_with_modulation(
    logits: torch.Tensor, modulation: torch.Tensor, labels: Optional[torch.Tensor] = None, return_type: str = 'p'
) -> torch.Tensor:

    log_probs_base = F.log_softmax(logits, dim=-1)
    log_probs_plus_modulation = log_probs_base + modulation
    # print("MODULATION")
    # print(modulation)
    new_log_probs = F.log_softmax(log_probs_plus_modulation, dim=-1)

    return return_or_gather_then_return(labels, new_log_probs, return_type)
    # log_probs_all_vocab = F.log_softmax(log_probs_plus_modulation, dim=-1)
    # if return_type == "all_vocab":
    #     return log_probs_all_vocab
    # log_probs_labels = log_probs_all_vocab.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # if return_type == "both":
    #     return log_probs_all_vocab, log_probs_labels
    # return log_probs_labels

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    else:
        return (tensor * mask).sum() / mask.sum()


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids
