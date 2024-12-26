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

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


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



class CTLLoss(nn.Module):
    """
    CTL Twist learning loss
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        curr_log_probs: torch.Tensor,
        base_action_log_probs: torch.Tensor
    ) -> torch.Tensor:
        positive_samples_term = 0.
        negative_samples_term = 0.
        # NOTE: this version of CTLLoss just uses reweighting (e.g. SIS version), no SMC resampling here (yet)

        print("CTL LOSS STUFF")

        # Sum across the t dimension to ensure we have the log prob of the FULL SEQUENCE
        base_action_seq_log_probs = base_action_log_probs.sum(dim=-1)[:, None]
        curr_seq_log_probs = curr_log_probs.sum(dim=-1)[:, None]
        print(base_action_seq_log_probs.shape)
        print(curr_seq_log_probs.shape)
        # TODO EXPECTED SHAPE FOR BOTH OF THE ABOVE: (batch_size, )

        # returns is the sum of future rewards, which is exactly the phi_t(s_{1:T}) that we want
        # basically the potential that we want to resample
        # ie returns are the phi_t(s_{1:T}) that we want to use
        # so we are going to resample (rather reweight/calculate weights here) according to sigma_t(s_{1:T}) for each t
        # then truncate each of those samples
        # then use the weights from each of those calculations of phi_t(s_{1:T})
        # the importance weights are p_0(s_{1:T}) phi_t(s_{1:T}) / q(s_{1:T})
        # thus the log weights will be log (ref_model(full_seq)) + log (returns) - log (curr_model(full_seq))
        # Only the middle term is different across different time steps, so I can calculate the first and last terms just once
        # then broadcast it according to the returns
        # Also make sure here that we are only taking the grad through the values and not through the policy here
        # Also, that the weights - we do not differentiate through the weights. The gradient should only be on log psi = values, not the returns
        # (Not differentiating through weights would be the same as in our previous TSMC paper/experiments)
        # Finally, we can sum up across t, as we want to learn for all t at the same time
        log_w_t_approx_sigma_samples = base_action_seq_log_probs + returns - curr_seq_log_probs
        # Is it log returns or just returns? I think just returns. Why? Because I already have defined
        # the reward in terms of a log phi, and then the other rewards are the log(q/p)
        # Which means that the sum of rewards, the returns, are just sum of intermediate reward + log phi
        # which in exp space, is the product of e^intermediate rewards and the final potential, which is
        # exactly the product of little phi and big phi as desired, e.g. in our appendix of the SMC paper
        # TODO write this logic out clearly, write out all the math in a document to make sure it adds up correctly
        # Btw it also just makes sense, just look at the structure of the two weights, you know they should be equal at the optimum
        # and clearly, the optimum is when values = returns
        log_psi_t_eval_list_proposal_samples = values # TODO ensure this is correct
        log_w_t_approx_pi_samples = base_action_seq_log_probs + values - curr_seq_log_probs


        print(log_psi_t_eval_list_proposal_samples.shape) # EXPECTED: (batch_size, seq_len)

        print(log_w_t_approx_sigma_samples.shape) # Expected: (batch_size, seq_len)
        print(log_w_t_approx_pi_samples.shape)

        # normalized_w_t_sigma_samples = F.softmax(
        #     log_w_t_approx_sigma_samples.detach())
        # log_psi_on_truncated_proposal_samples = values

        print("Wgt shapes")
        normalized_w_t_approx_sigma_samples = F.softmax(log_w_t_approx_sigma_samples.detach(), dim=0) # do softmax along the batch dimension
        print(normalized_w_t_approx_sigma_samples.shape)
        # EXPECTED: above has shape (batch_size, seq_len)
        positive_samples_term_new = normalized_w_t_approx_sigma_samples * log_psi_t_eval_list_proposal_samples
        print(positive_samples_term_new.shape)
        # EXPECTED: above has shape (batch_size, seq_len) - then can do masked mean on this

        # Try to do this batched instead of in for loop (verify that manual and batched give the same result)
        for i in range(log_w_t_approx_sigma_samples.shape[1]):
            positive_samples_term += (
                F.softmax(
                    log_w_t_approx_sigma_samples[:, i].detach()) @
                # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
                log_psi_t_eval_list_proposal_samples[:, i])
        positive_samples_term /= log_w_t_approx_sigma_samples.shape[1]

        print("Positive term check")
        print(positive_samples_term_new.sum(dim=-1).mean(dim=0))
        print(positive_samples_term) # TODO check, these should match each other
        print(positive_samples_term_new[:, 0] / returns[:, -1]) # TODO check: this should be a constant value; if not, investigate why not.



        # TODO INSTEAD OF ADDING THE DOT: have a 2-d array at the end and then do the masked mean on that

        normalized_w_t_approx_pi_samples = F.softmax(log_w_t_approx_pi_samples.detach(), dim=0) # do softmax along the batch dimension
        print(normalized_w_t_approx_pi_samples.shape)
        # EXPECTED: above has shape (batch_size, seq_len)
        negative_samples_term_new = normalized_w_t_approx_pi_samples * log_psi_t_eval_list_proposal_samples
        # EXPECTED: above has shape (batch_size, seq_len) - then can do masked mean on this

        # Try to do this batched instead of in for loop
        for i in range(log_w_t_approx_pi_samples.shape[1]):
            negative_samples_term += (
                F.softmax(
                    log_w_t_approx_pi_samples[:, i].detach()) @
                # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
                log_psi_t_eval_list_proposal_samples[:, i])
        negative_samples_term /= log_w_t_approx_pi_samples.shape[1]

        print("Negative term check")
        print(negative_samples_term_new.sum(dim=-1).mean(dim=0))
        print(negative_samples_term) # TODO check, these should match each other

        # This is the first term calculation, but really should do a similar kind of thing here as above
        loss = -(positive_samples_term_new - negative_samples_term_new)

        print(loss.shape)
        print(action_mask.shape)

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        # I guess mean is ok, just to keep things consistent. This does mean that I need ~10 to ~20x the learning rate
        # I had previously, in order to have the same results. So something like 1e-3 then... But can try 1e-4 just to be
        # consistent with what I have for the PPO critic loss...

        1/0 # TODO Don't remove until every little step along the way checked, and all makes sense.

        # print("--masked mean--")
        # print(masked_mean(loss, action_mask, dim=-1))
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
