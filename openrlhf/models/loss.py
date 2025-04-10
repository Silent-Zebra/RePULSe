from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

class REINFORCELoss(nn.Module):
    def __init__(self, baseline_type: Optional[str] = None, hardcoded_baseline: Optional[float] = None) -> None:
        super().__init__()
        self.baseline_type = baseline_type
        self.hardcoded_baseline = hardcoded_baseline

    def forward(
        self,
        log_probs: torch.Tensor,
        final_reward: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if len(log_probs.shape) == 3:
            reduce_mean_per_prompt = True
        elif len(log_probs.shape) == 2:
            reduce_mean_per_prompt = False
        else:
            raise NotImplementedError

        # print("REINFORCE LOSS INSPECTION")
        # print(final_reward.shape)
        # print(log_probs.shape)
        #
        # print("REWARDS")
        # print(final_reward)

        if self.baseline_type is not None:
            if self.baseline_type == "expectation":
                assert reduce_mean_per_prompt
                assert final_reward.shape[1] > 1 # this will do nothing if there is only 1 batch size/sample per prompt
                rewards_baseline = final_reward.mean(dim=1)  # mean along the batch dimension not the prompt dimension
                # print(rewards_baseline.shape)
                rewards_baseline = rewards_baseline.unsqueeze(1)
                # print(rewards_baseline.shape)

            elif self.baseline_type == "hardcoded":
                assert self.hardcoded_baseline is not None
                rewards_baseline = self.hardcoded_baseline

            else:
                raise NotImplementedError

            final_reward = final_reward - rewards_baseline

        # print("REWARDS AFTER BASELINE")
        # print(final_reward)
        # print("SHAPES")
        # print(final_reward.shape)
        # print(action_mask.shape)
        # print((log_probs * action_mask).shape)
        # print((log_probs * action_mask).sum(-1).shape)

        loss = (masked_mean(- log_probs, action_mask, -1) * final_reward).mean() # go from (prompts, batch_per_prompt, 1) to just (prompts, batch_per_prompt)
        # masked sum would be mathematically correct instead of masked mean, but is just a scalar shift for SGD, and for Adam, only affects early parts of training before the moments are learned
        # average may be more numerically stable over longer sequences (e.g., Adam would have larger moments for sum)

        # loss = (- (log_probs * action_mask).mean(-1) * final_reward.squeeze(-1)).mean() # final_reward.squeeze(-1) goes from (prompts, batch_per_prompt, 1) to just (prompts, batch_per_prompt)
        # loss2 = (masked_mean(- log_probs, action_mask, -1) * final_reward.squeeze(-1)).mean()
        # loss3 = masked_mean(- log_probs * final_reward, action_mask, -1).mean()
        # print(loss)
        # print(loss2)
        # print(loss3)
        # print(loss2 - loss)
        # print(loss3 - loss)
        # print(loss3 - loss2)

        return loss

class NegTrainingLoss(nn.Module):
    def __init__(self, alpha=0.5, baseline_type: Optional[str] = None, hardcoded_baseline: Optional[float] = None) -> None:
        super().__init__()
        self.reinforce_loss_fn = REINFORCELoss(baseline_type=baseline_type, hardcoded_baseline=hardcoded_baseline)
        self.alpha = alpha

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_neg: torch.Tensor,
        final_reward: torch.Tensor,
        normalized_w_t_approx_sigma_samples: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action_mask_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        reinforce_loss = self.reinforce_loss_fn(log_probs, final_reward, action_mask)

        # print("REWARDS AFTER BASELINE")
        # print(log_probs_neg.shape)
        # print(final_reward.shape)
        #
        # print("WEIGHTS SHAPE")
        # print(normalized_w_t_approx_sigma_samples.shape)
        # print(normalized_w_t_approx_sigma_samples)


        # loss = log_probs_neg * normalized_w_t_approx_sigma_samples.detach().unsqueeze(-1) # Negative training loss: just push down on log probs. Therefore reduce loss: reduce log probs
        # # TODO check weighting is correct, also normalize with softmax if necessary
        #
        # loss = masked_mean(loss, action_mask, dim=-1).mean()
        # loss2 = ((log_probs_neg * action_mask).mean(-1) * normalized_w_t_approx_sigma_samples.detach()).mean()
        # print(loss)
        # print(loss2)
        # print(loss2 - loss)


        # Just reduce log probs on all of the negative samples (approximate sigma samples)
        # The multiplication by normalized_w_t_approx_sigma_samples is because we have approximate sigma samples
        # And since I didn't do resampling, just doing reweighting, then normalized_w_t_approx_sigma_samples are the weights
        # So we need to multiply by those weights to get the approximate sigma samples


        # loss_old = ((log_probs_neg * action_mask_neg).mean(-1) * normalized_w_t_approx_sigma_samples.detach()).mean() # mean instead of masked mean has issues - problems with reweighting on the masked things
        loss = (masked_mean(log_probs_neg, action_mask_neg, dim=-1) * normalized_w_t_approx_sigma_samples.detach()).mean()
        # print(loss - loss_old)

        # print("NEG TRAIN LOSS")
        # print(loss)
        # print(masked_mean(log_probs_neg, action_mask_neg, dim=-1))
        # print(normalized_w_t_approx_sigma_samples.detach())
        #
        # print("FINAL LOSS")
        # print(self.alpha * loss)
        # print((1 - self.alpha) * reinforce_loss + self.alpha * loss)


        # Weighting here controls how much to emphasize the negative training (reduce probability on negative samples) loss vs. the standard REINFORCE/RL loss
        return (1 - self.alpha) * reinforce_loss + self.alpha * loss


class NegREINFORCELoss(nn.Module):
    def __init__(
        self, alpha=0.5, baseline_type: Optional[str] = None, hardcoded_baseline: Optional[float] = None,
        baseline_type_neg: Optional[str] = None, hardcoded_baseline_neg: Optional[float] = None
    ) -> None:
        super().__init__()
        self.reinforce_loss_fn = REINFORCELoss(baseline_type=baseline_type, hardcoded_baseline=hardcoded_baseline)
        self.reinforce_loss_fn_neg = REINFORCELoss(baseline_type=baseline_type_neg, hardcoded_baseline=hardcoded_baseline_neg)

        self.alpha = alpha

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_neg: torch.Tensor,
        rewards: torch.Tensor,
        rewards_neg: torch.Tensor,
        normalized_w_t_approx_sigma_samples: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action_mask_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        reinforce_loss = self.reinforce_loss_fn(log_probs, rewards, action_mask)

        log_probs_neg *= normalized_w_t_approx_sigma_samples.detach().unsqueeze(-1)

        # print(log_probs_neg.shape)
        # print(normalized_w_t_approx_sigma_samples.shape)
        # print(normalized_w_t_approx_sigma_samples.detach().unsqueeze(-1).shape)
        # print(normalized_w_t_approx_sigma_samples)
        # 1/0

        neg_reinforce_loss = self.reinforce_loss_fn_neg(log_probs_neg, rewards_neg, action_mask_neg)


        return (1 - self.alpha) * reinforce_loss + self.alpha * neg_reinforce_loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # print("PPO ACTOR LOSS STUFF")
        # print(log_probs.shape)
        # print(log_probs)
        # print(old_log_probs.shape)
        # print(old_log_probs)
        # print(advantages.shape)
        # print(advantages)
        # print(action_mask.shape)
        # print(action_mask)


        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        # print("RATIO")
        # print(ratio)
        # print("ADVANTAGES")
        # print(advantages)
        #
        # print("ACTOR LOSS DETAILS")
        # print(surr1)
        # print(surr2)
        # print(loss)
        # print(loss.abs())
        # print(masked_mean(loss.abs(), action_mask, dim=-1).mean())
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        # print(loss)
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        # print("--VALUE LOSS--")
        # print("--values--")
        # print(values)
        # print("--returns--")
        # print(returns)
        # print("--loss--")
        # print(loss)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        # print("--masked mean--")
        # print(masked_mean(loss, action_mask, dim=-1))
        return 0.5 * loss


def get_positive_and_negative_weights_detached(base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples):

    # Now let's just do the standard CTL loss... all we have is just the p * phi / q for reweighting here...
    # Sum across the t dimension to ensure we have the log prob of the FULL SEQUENCE
    log_w_t_approx_sigma_samples = get_positive_weights_detached(base_action_log_probs, curr_log_probs, final_reward)
    # print(log_psi_t_eval_list_proposal_samples)
    # print(log_psi_t_eval_list_proposal_samples.shape)
    # print(base_action_log_probs.cumsum(dim=1).shape)
    # print(curr_log_probs.cumsum(dim=1).shape)
    log_w_t_approx_pi_samples = base_action_log_probs.cumsum(
        dim=1) + log_psi_t_eval_list_proposal_samples - curr_log_probs.cumsum(
        dim=1)  # because here our IS weights are p * psi in numerator, as in our previous paper, divided by q. And with values = log psi, and us working in log space, this is what we get. Note that we are reweighting according to p(s_1:t) psi_t(s_1:t) / q(s_1:t) which is why we have cumsum
    log_w_t_approx_pi_samples = log_w_t_approx_pi_samples.detach()
    # print(log_psi_t_eval_list_proposal_samples.shape) # EXPECTED: (batch_size, seq_len)
    # print("Log Wgt shapes")
    # print(log_w_t_approx_sigma_samples.shape) # Expected: (batch_size)
    # print(log_w_t_approx_pi_samples.shape) # Expected: (batch_size, seq_len)
    # normalized_w_t_sigma_samples = F.softmax(
    #     log_w_t_approx_sigma_samples.detach())
    # log_psi_on_truncated_proposal_samples = values
    # print("Wgt shapes")
    if len(log_w_t_approx_sigma_samples.shape) > 1:
        raise NotImplementedError # dim 0 will be wrong if you use this with batch over diff prompts.

    normalized_w_t_approx_sigma_samples = F.softmax(log_w_t_approx_sigma_samples,
                                                    dim=0)  # do softmax along the batch dimension, which is -1
    # print(normalized_w_t_approx_sigma_samples.shape)
    # EXPECTED: above has shape (batch_size)


    return log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples


def get_positive_weights_detached(base_action_log_probs, curr_log_probs, final_reward):
    log_w_t_approx_sigma_samples = base_action_log_probs.sum(dim=-1) + final_reward - curr_log_probs.sum(
        dim=-1)  # why this: well, the target is base * phi, then denom for IS is q.
    log_w_t_approx_sigma_samples = log_w_t_approx_sigma_samples.detach()

    return log_w_t_approx_sigma_samples

def get_normalized_positive_weights_detached(base_action_log_probs, curr_log_probs, final_reward, batch_dim=0):
    log_w_t_approx_sigma_samples = get_positive_weights_detached(base_action_log_probs, curr_log_probs, final_reward)
    assert len(log_w_t_approx_sigma_samples.shape) <= 2 # Covers (batch) and (prompts, batch) shapes, but not others
    normalized_w_t_approx_sigma_samples = F.softmax(log_w_t_approx_sigma_samples,
                                                    dim=-1)  # do softmax along the batch dimension

    return normalized_w_t_approx_sigma_samples



def get_positive_and_negative_weights_detached_incremental(base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples):
    log_p_1_to_t_psi_1_to_t = base_action_log_probs.cumsum(dim=1) + log_psi_t_eval_list_proposal_samples
    log_w_t = 0
    negative_weights = []

    # log_w_t_approx_pi_samples_ref = base_action_log_probs.cumsum(
    #     dim=1) + log_psi_t_eval_list_proposal_samples - curr_log_probs.cumsum(
    #     dim=1)

    for i in range(base_action_log_probs.shape[-1]):
        if i == 0:
            incremental_w_t = log_p_1_to_t_psi_1_to_t[:, 0] - curr_log_probs[:, 0]
            log_w_t += incremental_w_t # BE CAREFUL - THIS IN-PLACE OPERATION MODIFIES MUTABLE STRUCTURES LIKE LISTS; THIS IS WHY JAX DOESN'T ALLOW THIS KIND OF STUFF
        elif i == base_action_log_probs.shape[-1] - 1:
            incremental_w_t = base_action_log_probs.cumsum(dim=1)[:, -1] + final_reward - curr_log_probs[:, i] - log_p_1_to_t_psi_1_to_t[:, i - 1]
            positive_total_weight = incremental_w_t + log_w_t
            negative_incremental_w_t = log_p_1_to_t_psi_1_to_t[:, i] - curr_log_probs[:, i] - log_p_1_to_t_psi_1_to_t[:, i - 1]
            log_w_t += negative_incremental_w_t # BE CAREFUL - THIS IN-PLACE OPERATION MODIFIES MUTABLE STRUCTURES LIKE LISTS; THIS IS WHY JAX DOESN'T ALLOW THIS KIND OF STUFF
        else:
            incremental_w_t = log_p_1_to_t_psi_1_to_t[:, i] - curr_log_probs[:, i] - log_p_1_to_t_psi_1_to_t[:, i - 1]
            log_w_t += incremental_w_t # BE CAREFUL - THIS IN-PLACE OPERATION MODIFIES MUTABLE STRUCTURES LIKE LISTS; THIS IS WHY JAX DOESN'T ALLOW THIS KIND OF STUFF

        # print(f'iter {i}')
        # print(log_w_t)
        # print(log_w_t_approx_pi_samples_ref[:, i])
        # print(torch.abs(log_w_t_approx_pi_samples_ref[:, i] - log_w_t).mean())

        negative_weights.append(log_w_t.detach().clone()) # CLONE IS VERY IMPORTANT HERE
    normalized_w_t_approx_sigma_samples = F.softmax(positive_total_weight, dim=0).detach()  # do softmax along the batch dimension

    # # DETACH ON WEIGHTS IS IMPORTANT FOR THE RIGHT GRADIENTS
    # print("final comparison")
    negative_weights = torch.stack(negative_weights, dim=1).detach()
    # print(negative_weights - log_w_t_approx_pi_samples_ref)
    # print(torch.abs(negative_weights - log_w_t_approx_pi_samples_ref).mean())

    # print("Final val 2")
    # print(negative_weights)

    return negative_weights, normalized_w_t_approx_sigma_samples


class CTLLoss(nn.Module):
    """
    CTL Twist learning loss
    """

    def __init__(self, no_second_term=False) -> None:
        super().__init__()
        self.no_second_term = no_second_term

    def forward(
        self,
        values: torch.Tensor,
        final_reward: torch.Tensor,
        action_mask: torch.Tensor,
        curr_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: this version of CTLLoss just uses reweighting (e.g. SIS version), no SMC resampling here (yet)
        # Note that if you were to do resampling, we would need to figure out how to deal with varying sequence lengths (when EOS generated)
        # Right now, the code does right padding (replay buffer swaps padding from left to right), which I think is a big problem for resampling
        # It's fine for SIS, because the log probs are invariant to padding as long as you pass in the right attention mask
        # But for intermediate resampling, I imagine we probably want left padding instead. And then there's the question of what happens after EOS is generated
        # If you resample a sequence that has EOS, is it just stuck like that forever afterwards?
        # Should investigate how people doing SMC for LLM (maybe Lew et al also) deal with this issue, but that will be for later when doing resampling

        # print(values.shape)
        if len(values.shape) == 3:
            reduce_mean_per_prompt = True
        elif len(values.shape) == 2:
            reduce_mean_per_prompt = False
        else:
            raise NotImplementedError

        # print(reduce_mean_per_prompt)

        # print("CTLLOSS INSPECTION")
        # print(action_mask.shape)
        # print(action_mask)
        # print(curr_log_probs.shape)
        # print(curr_log_probs)
        # print(curr_log_probs.sum(-1))
        # print(base_action_log_probs.shape)
        # print(base_action_log_probs)
        # print(values.shape)
        # print(values)

        # Set log probs of padding tokens to be 0, so that when they are added, they don't affect anything.
        # curr_log_probs *= action_mask # this one already handled by the replay buffer I believe, so this is redundant
        base_action_log_probs *= action_mask
        values *= action_mask # This should also be redundant since the masked mean at the end should take care of the values; values (log_psi) should be 0 after the final masked mean and have 0 gradient there for tokens after EOS
        # But I'm leaving the above just to be safe; TODO later can test to ensure this is the case.

        # print("After mask")
        # print(curr_log_probs.sum(-1))

        if action_mask.shape[-1] > 100: # Where EOS may start to come into play.
            raise Exception("CHECK THE EOS AND PADDING AND ACTION MASK CAREFULLY, ENSURE IT WORKS AS EXPECTED. Should work, but just confirm and test")

        if reduce_mean_per_prompt:
            # This version is for batching over different prompts
            # Use vmap to compute weights for all prompts at once
            batched_get_weights = torch.func.vmap(get_positive_and_negative_weights_detached, in_dims=0)
            log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples = batched_get_weights(
                base_action_log_probs,
                curr_log_probs,
                final_reward,
                values
            )

            # Compute terms using the vectorized weights
            positive_samples_term = normalized_w_t_approx_sigma_samples.unsqueeze(-1) * values
            normalized_w_t_approx_pi_samples = F.softmax(log_w_t_approx_pi_samples, dim=1)
            negative_samples_term = normalized_w_t_approx_pi_samples * values

            # print("weight inspection")
            # print(log_w_t_approx_pi_samples)
            # print(normalized_w_t_approx_pi_samples)

            # print('wgts')
            # print(normalized_w_t_approx_sigma_samples)
            # print(normalized_w_t_approx_pi_samples)
            #
            # print('comparison')
            # print(torch.exp(curr_log_probs - base_action_log_probs))
            #
            # print('shapes')
            # print(action_mask.shape)
            # print(positive_samples_term.shape)
            # print(negative_samples_term.shape)
            # print(log_w_t_approx_pi_samples.shape)
            # print(normalized_w_t_approx_sigma_samples.shape)


            if self.no_second_term:
                loss = - positive_samples_term
            else:
                loss = -(positive_samples_term - negative_samples_term)
            # print(loss.shape)

            loss = masked_mean(loss, action_mask, dim=-1).mean()
            # print(loss)

            # 1/0
            # TODO March 8 after this, try running the previous experiments (custom single prompt) except now just use
            # these vmapped losses instead. First try exact reproduction, but use format below
            # Try: even if prompt repeated, put it into batches, and softmax/weights over each of those individual batches
            # As you set the prompt batch size to 1 and n_samples_per_prompt to previous batch size, it should reproduce
            # As you set prompt batch size bigger and n_samples_per_prompt to smaller, the loss should get more and more noisy and worse performance. 1 sample per prompt should totally fail.
            # TODO LATER test also the SIXO loss formulation


            return loss

        # Values = log_psi in the twist formulation
        # final_reward = log phi
        # curr_log_probs = log q
        # base_action_log_probs = log p_0
        # Therefore, to calculate positive weights, we just need p * phi / q (in log terms, log p + log phi - log q)
        # For negative weights, we need p * psi / q (in log space, log p + log psi - log q)

        # print("CTL ACTOR LOSS FUNCTION")
        # print(final_reward.shape)
        # print(base_action_log_probs.sum(dim=-1).shape)

        # print("CTL LOSS STUFF")
        log_psi_t_eval_list_proposal_samples = values
        log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples = get_positive_and_negative_weights_detached(
            base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples)

        # log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples = get_positive_and_negative_weights_detached_incremental(
        #     base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples)
        # TODO REMOVE ABOVE LATER COMPARISON ONLY
        # print("FINAL")
        # print(torch.abs(log_w_t_approx_pi_samples2 - log_w_t_approx_pi_samples))
        # print(torch.abs(log_w_t_approx_pi_samples2 - log_w_t_approx_pi_samples).mean())
        # print(torch.abs(normalized_w_t_approx_sigma_samples2 - normalized_w_t_approx_sigma_samples).mean())
        # 1/0

        positive_samples_term_new = normalized_w_t_approx_sigma_samples[:, None] * log_psi_t_eval_list_proposal_samples
        # print(positive_samples_term_new.shape)
        # EXPECTED: above has shape (batch_size, seq_len) - then can do masked mean on this

        normalized_w_t_approx_pi_samples = F.softmax(log_w_t_approx_pi_samples, dim=0) # do softmax along the batch dimension
        # print(normalized_w_t_approx_pi_samples.shape)
        # EXPECTED: above has shape (batch_size, seq_len)
        negative_samples_term_new = normalized_w_t_approx_pi_samples * log_psi_t_eval_list_proposal_samples
        # EXPECTED: above has shape (batch_size, seq_len) - then can do masked mean on this

        # Try to do this batched instead of in for loop
        # for i in range(log_w_t_approx_pi_samples.shape[1]):
        #     negative_samples_term += (
        #         F.softmax(
        #             log_w_t_approx_pi_samples[:, i], dim=0) @ log_psi_t_eval_list_proposal_samples[:, i])
        #
        #         # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
        # negative_samples_term /= log_w_t_approx_pi_samples.shape[1]

        # TODO Consider doing this
        # Why? Because: multiplying by normalized weights, we are already reducing each value. For the weighted mean, we multiply by weights, then add up. So if I multiply by weights, then do mean, I'm dividing by the batch size twice, which seems undesirable
        # Of course here for CTL it doesn't really matter, because this is just a constant rescaling of the loss which can be absorbed into the learning rate. But still, maybe is good to just keep it consistent with the math and with the previous implementation
        # print(positive_samples_term_new.shape[0])
        # print(negative_samples_term_new.shape[0])
        # positive_samples_term_new *= positive_samples_term_new.shape[0]
        # negative_samples_term_new *= negative_samples_term_new.shape[0]
        # Arguably this makes things worse; arguably you would rather not rescale e.g. * 100 and then have a 100x lower lr, you'd rather just have the 100x higher lr and avoid the *100/100 calculation which maybe loses precision
        # ACTUALLY all of this logic is not correct. It is correct for simple SGD, but for Adam, since it scales by moments, in the limit,
        # you actually get the same behaviour regardless of sum or avg or scaling by a factor of 100.
        # HOWEVER you may have different behaviour at the beginning, before Adam has learned the moments well
        # So with sum instead of avg you may have more instability and more aggressive gradient updates at the start of training
        # Maybe this is bad for SIXO? Or rather, less aggressive updates/scaled down values at the beginning is better?

        # print("Negative term check")
        # print(negative_samples_term_new.sum(dim=0).mean(dim=-1))
        # print(negative_samples_term) # check, these should match each other

        if self.no_second_term:
            loss = - positive_samples_term_new
        else:
            loss = -(positive_samples_term_new - negative_samples_term_new)

        # print(loss.shape)
        # print(action_mask.shape)

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        # loss = masked_mean(loss, action_mask, dim=-1).sum()

        # print("--masked mean--")
        # print(masked_mean(loss, action_mask, dim=-1))
        return loss




class MixedCTLValueLoss(nn.Module):
    def __init__(self, clip_eps: float = None, alpha: float = 0.5) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.alpha = alpha
        self.value_loss = ValueLoss(clip_eps)
        self.ctl_loss = CTLLoss()

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        curr_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor,
        final_reward: torch.Tensor
    ) -> torch.Tensor:
        ctl_loss = self.ctl_loss(values, final_reward, action_mask, curr_log_probs, base_action_log_probs)
        mse_loss = self.value_loss(values, old_values, returns, action_mask)
        return self.alpha * ctl_loss + (1 - self.alpha) * mse_loss


class SIXOLoss(nn.Module):
    """
    SIXO Twist learning loss
    """

    def __init__(self, approx_neg: bool = False) -> None:
        super().__init__()
        self.approx_neg = approx_neg

    def forward(
        self,
        values: torch.Tensor,
        final_reward: torch.Tensor,
        action_mask: torch.Tensor,
        curr_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor,
        values_on_base_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.approx_neg:
            assert values_on_base_samples is None
        else:
            assert values_on_base_samples is not None

        # print(values.shape)
        if len(values.shape) == 3:
            reduce_mean_per_prompt = True
        elif len(values.shape) == 2:
            reduce_mean_per_prompt = False
        else:
            raise NotImplementedError

        # print(reduce_mean_per_prompt)

        # NOTE: this version of SIXOLoss just uses reweighting (e.g. SIS version), no SMC resampling here (yet)
        # Note that if you were to do resampling, we would need to figure out how to deal with varying sequence lengths (when EOS generated)
        # Right now, the code does right padding (replay buffer swaps padding from left to right), which I think is a big problem for resampling
        # It's fine for SIS, because the log probs are invariant to padding as long as you pass in the right attention mask
        # But for intermediate resampling, I imagine we probably want left padding instead. And then there's the question of what happens after EOS is generated
        # If you resample a sequence that has EOS, is it just stuck like that forever afterwards?
        # Should investigate how people doing SMC for LLM (maybe Lew et al also) deal with this issue, but that will be for later when doing resampling

        # Set log probs of padding tokens to be 0, so that when they are added, they don't affect anything.
        # curr_log_probs *= action_mask # this one already handled by the replay buffer I believe, so this is redundant
        base_action_log_probs *= action_mask
        values *= action_mask  # This should also be redundant since the masked mean at the end should take care of the values; values (log_psi) should be 0 after the final masked mean and have 0 gradient there for tokens after EOS
        # But I'm leaving the above just to be safe; TODO later can test to ensure this is the case.

        # print("After mask")
        # print(curr_log_probs.sum(-1))

        if action_mask.shape[-1] > 100:  # Where EOS may start to come into play.
            raise Exception("CHECK THE EOS AND PADDING AND ACTION MASK CAREFULLY, ENSURE IT WORKS AS EXPECTED. Should work, but just confirm and test")


        if reduce_mean_per_prompt:
            # This version is for batching over different prompts
            # Use vmap to compute weights for all prompts at once
            batched_get_weights = torch.func.vmap(get_normalized_positive_weights_detached, in_dims=0)
            # log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples = batched_get_weights(
            #     base_action_log_probs,
            #     curr_log_probs,
            #     final_reward,
            #     values
            # )
            normalized_w_t_approx_sigma_samples = batched_get_weights(
                base_action_log_probs,
                curr_log_probs,
                final_reward,
                # values
            )

            # Compute positive term with batched weights
            positive_samples_term = normalized_w_t_approx_sigma_samples.unsqueeze(-1) * F.logsigmoid(values)

            # print('shapes')
            # print(action_mask.shape)
            # print(positive_samples_term.shape)
            # print(log_w_t_approx_pi_samples.shape)
            # print(normalized_w_t_approx_sigma_samples.shape)
            # if values_on_base_samples is not None:
            #     print(values_on_base_samples.shape)

            if self.approx_neg:
                # For approximate negative samples, compute weights based on p/q ratio for each prompt
                log_w_t_approx_p_samples = base_action_log_probs.sum(dim=-1) - curr_log_probs.sum(dim=-1)
                log_w_t_approx_p_samples = log_w_t_approx_p_samples.detach()

                # print("approx neg inspection")
                # print(log_w_t_approx_p_samples.shape)

                # Normalize weights per prompt batch
                normalized_w_t_approx_p_samples = F.softmax(log_w_t_approx_p_samples, dim=1)  # softmax over samples within each prompt

                # print(normalized_w_t_approx_p_samples.shape)

                negative_samples_term = normalized_w_t_approx_p_samples.unsqueeze(-1) * torch.log(1 - F.sigmoid(values))

                # print(negative_samples_term.shape)


            else:
                # For exact negative samples, use provided base samples
                negative_samples_term = torch.log(1 - F.sigmoid(values_on_base_samples))
                # Average across samples within each prompt batch
                negative_samples_term = negative_samples_term / negative_samples_term.shape[1]

            # print(negative_samples_term.shape)


            # Compute final loss with negative term
            loss = -(positive_samples_term + negative_samples_term)
            
            # Apply action mask and sum across prompts
            # loss = masked_mean(loss, action_mask, dim=-1).sum()
            loss = masked_mean(loss, action_mask, dim=-1).mean()

            return loss


        # print("SIXO LOSS STUFF")
        # First step is the same as in CTL; get the approx sigma samples based on p * phi / q on the FULL SEQUENCE then truncating
        # Sum across the t dimension to ensure we have the log prob of the FULL SEQUENCE
        # Again I use q as the proposal and do SIS reweighting
        # print(final_reward.shape)
        # print(base_action_log_probs.sum(dim=-1).shape)

        log_psi_t_eval_list_proposal_samples = values
        # log_w_t_approx_pi_samples, normalized_w_t_approx_sigma_samples = get_positive_and_negative_weights_detached(
        #     base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples)
        normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
            base_action_log_probs, curr_log_probs, final_reward)

        # _, normalized_w_t_approx_sigma_samples = get_positive_and_negative_weights_detached_incremental(base_action_log_probs, curr_log_probs, final_reward, log_psi_t_eval_list_proposal_samples)
        # # TODO REMOVE ABOVE LATER COMPARISON ONLY

        # print("HIHI")
        # print(normalized_w_t_approx_sigma_samples2)
        # print(normalized_w_t_approx_sigma_samples2 - normalized_w_t_approx_sigma_samples)
        # print(torch.abs(normalized_w_t_approx_sigma_samples2 - normalized_w_t_approx_sigma_samples).mean())
        # 1/0

        # positive_samples_term_new = normalized_w_t_approx_sigma_samples[:,
        #                         None] * F.logsigmoid(values)
        #
        # log_w_t_approx_sigma_samples = base_action_log_probs.sum(
        #     dim=-1) + final_reward - curr_log_probs.sum(
        #     dim=-1)  # why this: well, the target is base * phi, then denom for IS is q.
        # log_w_t_approx_sigma_samples = log_w_t_approx_sigma_samples.detach()
        #
        # normalized_w_t_approx_sigma_samples = F.softmax(
        #     log_w_t_approx_sigma_samples,
        #     dim=0)  # do softmax along the batch dimension

        positive_samples_term = normalized_w_t_approx_sigma_samples[:, None] * F.logsigmoid(log_psi_t_eval_list_proposal_samples)

        # print(F.logsigmoid(values).shape) # Expected (batch, seq_len)

        # print(positive_samples_term.shape[0]) # Expected (batch)

        # print(positive_samples_term.shape)
        # EXPECTED: above has shape (batch_size, seq_len) - then can do masked mean on this

        if self.approx_neg:
            log_w_t_approx_p_samples = base_action_log_probs.sum(
                dim=-1) - curr_log_probs.sum(
                dim=-1)  # target p, denom for IS is q.
            log_w_t_approx_p_samples = log_w_t_approx_p_samples.detach()

            normalized_w_t_approx_p_samples = F.softmax(
                log_w_t_approx_p_samples,
                dim=0)  # do softmax along the batch dimension
            negative_samples_term = normalized_w_t_approx_p_samples[:,
                                    None] * torch.log(1 - F.sigmoid(values))
        else: # use exact p samples

            # print("positive weights")
            # print(normalized_w_t_approx_sigma_samples[:,None])
            # print(normalized_w_t_approx_sigma_samples[:,None].shape)
            # print("logisgmoid values")
            # print(F.logsigmoid(values))
            # print(F.logsigmoid(values).shape)
            # print("positive sample terms")
            # print(positive_samples_term)
            # print(positive_samples_term.shape)

            negative_samples_term = torch.log(1 - F.sigmoid(values_on_base_samples))

            # print("negative sample terms")
            # print(negative_samples_term)
            # print(negative_samples_term.shape)

            # positive_samples_term *= positive_samples_term.shape[0]
            # Should actually do the above on CTL too (for both on CTL). Why? Because: multiplying by normalized weights, we are already reducing each value. For the weighted mean, we multiply by weights, then add up. So if I multiply by weights, then do mean, I'm dividing by the batch size twice, which is undesirable
            # The negative term here doesn't need this multiplication since it's not being multiplied by the normalized weights

            negative_samples_term /= negative_samples_term.shape[0]
            # Alternatively: I can do this to make things the same... now this is consistent with mean on top of mean (which I believe does too much dividing... but oh well.
            # At least this now makes sixoloss and sixoloss using approx p samples based on IS reweighting of q samples, have the same scale

            # print("negative sample terms 2")
            # print(negative_samples_term)
            # print(negative_samples_term.shape)

        # print("Negative term check")
        # print(negative_samples_term_new.sum(dim=0).mean(dim=-1))
        # print(negative_samples_term) # check, these should match each other

        # This is the first term calculation, but really should do a similar kind of thing here as above
        loss = - (positive_samples_term + negative_samples_term) # see my new derivation; the KL divergence/loss has the negative term

        # print(loss.shape)
        # print(action_mask.shape)

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        # loss = masked_mean(loss, action_mask, dim=-1).sum()

        return loss.float()


class DPGLoss(nn.Module):
    """
    DPG policy learning loss
    """

    def __init__(self) -> None:
        super().__init__()


    def forward(
        self,
        values: torch.Tensor,
        final_reward: torch.Tensor,
        action_mask: torch.Tensor,
        curr_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor,
        log_psi_all_vocab: torch.Tensor,
        base_action_log_probs_all_vocab: torch.Tensor,
    ) -> torch.Tensor:
        if len(values.shape) == 3:
            reduce_mean_per_prompt = True
        elif len(values.shape) == 2:
            reduce_mean_per_prompt = False
        else:
            raise NotImplementedError

        # raise NotImplementedError # not yet done

        # Set log probs of padding tokens to be 0, so that when they are added, they don't affect anything.
        # curr_log_probs *= action_mask # this one already handled by the replay buffer I believe, so this is redundant
        base_action_log_probs *= action_mask
        values *= action_mask # This should also be redundant since the masked mean at the end should take care of the values; values (log_psi) should be 0 after the final masked mean and have 0 gradient there for tokens after EOS
        # But I'm leaving the above just to be safe; TODO later can test to ensure this is the case.

        if action_mask.shape[-1] > 100: # Where EOS may start to come into play.
            raise Exception("CHECK THE EOS AND PADDING AND ACTION MASK CAREFULLY, ENSURE IT WORKS AS EXPECTED. Should work, but just confirm and test")

        if reduce_mean_per_prompt:
            # This version is for batching over different prompts
            # Use vmap to compute weights for all prompts at once
            batched_get_weights = torch.func.vmap(get_normalized_positive_weights_detached, in_dims=0)
            normalized_w_t_approx_sigma_samples = batched_get_weights(
                base_action_log_probs,
                curr_log_probs,
                final_reward,
                # values
                return_normalized=True
            )

            # # Compute terms using the vectorized weights
            # positive_samples_term = normalized_w_t_approx_sigma_samples.unsqueeze(-1) * values
            #
            # normalized_p_psi_all_vocab = torch.softmax(base_action_log_probs_all_vocab + log_psi_all_vocab, dim=-1)
            # # get all logits - a bit annoying since you have to modify the forward calls in both actor and actor_custom to produce all logits, and then do the sum/reduce over them
            # negative_samples_term = (
            #     normalized_p_psi_all_vocab * log_psi_all_vocab).sum(
            #     axis=-1)
            #
            # loss = -(positive_samples_term - negative_samples_term)
            #
            # loss = masked_mean(loss, action_mask, dim=-1).sum()
            #
            # return loss
        else:
            normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
                base_action_log_probs, curr_log_probs, final_reward)

        log_psi_t_eval_list_proposal_samples = values

        # print(normalized_w_t_approx_sigma_samples.shape)
        # print(normalized_w_t_approx_sigma_samples[:, None].shape)
        # print(log_psi_t_eval_list_proposal_samples.shape)


        positive_samples_term = log_psi_t_eval_list_proposal_samples


        normalized_p_psi_all_vocab = torch.softmax(base_action_log_probs_all_vocab + log_psi_all_vocab, dim=-1).detach() # IMPORTANT: need not to propagate through weights

        # get all logits - a bit annoying since you have to modify the forward calls in both actor and actor_custom to produce all logits, and then do the sum/reduce over them
        negative_samples_term = (
            normalized_p_psi_all_vocab * log_psi_all_vocab).sum(
            axis=-1)  # The log psi is where we'll get the gradient (grad Q), and then the sum does the expectation over q(s_t | s_1:t-1)
        # Mean along the time dimension, again we can debate if we want to use sum. Just be consistent, that's the most important.


        loss = -normalized_w_t_approx_sigma_samples.unsqueeze(-1) * (positive_samples_term - negative_samples_term)

        # loss = masked_mean(loss, action_mask, dim=-1).sum()
        loss = masked_mean(loss, action_mask, dim=-1).mean()

        return loss




class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
