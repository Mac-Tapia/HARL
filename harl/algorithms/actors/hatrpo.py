"""HATRPO algorithm."""

import numpy as np
import torch
from harl.utils.envs_tools import check
from harl.utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence,
)
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.stochastic_policy import StochasticPolicy


class HATRPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HATRPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        assert (
            act_space.__class__.__name__ != "MultiDiscrete"
        ), "only continuous and discrete action space is supported by HATRPO."
        super(HATRPO, self).__init__(args, obs_space, act_space, device)

        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]

    @staticmethod
    def _safe_float(value, default=0.0):
        """Return a finite Python float for logger/export code."""
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().cpu().item()
        elif isinstance(value, np.ndarray):
            value = float(np.nanmean(value)) if value.size else default
        else:
            value = float(value)
        return value if np.isfinite(value) else default

    @staticmethod
    def _skipped_update(dist_entropy=0.0, ratio=1.0, reason="nonfinite"):
        return {
            "kl": 0.0,
            "policy_loss": 0.0,
            "loss_improve": 0.0,
            "expected_improve": 0.0,
            "dist_entropy": HATRPO._safe_float(dist_entropy),
            "ratio": HATRPO._safe_float(ratio, default=1.0),
            "actor_grad_norm": 0.0,
            "trpo_step_norm": 0.0,
            "trpo_update_accepted": 0.0,
            "trpo_skipped_nonfinite": 1.0 if reason == "nonfinite" else 0.0,
            "trpo_skipped_no_signal": 1.0 if reason == "no_signal" else 0.0,
        }

    def update(self, sample):
        """Update actor networks.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            kl: (torch.Tensor) KL divergence between old and new policy.
            loss_improve: (np.float32) loss improvement.
            expected_improve: (np.ndarray) expected loss improvement.
            dist_entropy: (torch.Tensor) action entropies.
            ratio: (torch.Tensor) ratio between new and old policy.
        """

        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        old_action_log_probs_batch = torch.nan_to_num(
            old_action_log_probs_batch, nan=0.0, posinf=20.0, neginf=-20.0
        )
        adv_targ = torch.nan_to_num(adv_targ, nan=0.0, posinf=0.0, neginf=0.0)
        adv_targ = torch.clamp(adv_targ, min=-10.0, max=10.0)
        factor_batch = torch.nan_to_num(factor_batch, nan=1.0, posinf=1.0, neginf=1.0)
        factor_batch = torch.clamp(factor_batch, min=0.05, max=20.0)
        if self.use_policy_active_masks and active_masks_batch.sum() <= 1e-6:
            return self._skipped_update(reason="no_signal")

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        if not (
            torch.isfinite(action_log_probs).all()
            and torch.isfinite(dist_entropy).all()
        ):
            return self._skipped_update(dist_entropy=dist_entropy)

        # actor update
        log_ratio = torch.clamp(
            action_log_probs - old_action_log_probs_batch,
            min=-20.0,
            max=20.0,
        )
        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(log_ratio),
            dim=-1,
            keepdim=True,
        )
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=20.0, neginf=0.05)
        if self.use_policy_active_masks:
            denom = torch.clamp(active_masks_batch.sum(), min=1.0)
            loss = (
                torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / denom
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ, dim=-1, keepdim=True
            ).mean()
        if not torch.isfinite(loss):
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio)

        loss_grad = torch.autograd.grad(
            loss, self.actor.parameters(), allow_unused=True
        )
        loss_grad = flat_grad(loss_grad)
        if loss_grad.numel() == 0:
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio, reason="no_signal")
        loss_grad = torch.nan_to_num(loss_grad, nan=0.0, posinf=0.0, neginf=0.0)
        actor_grad_norm = torch.norm(loss_grad)
        if (not torch.isfinite(actor_grad_norm)) or actor_grad_norm <= 1e-12:
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio, reason="no_signal")

        step_dir = conjugate_gradient(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            loss_grad.data,
            nsteps=10,
            device=self.device,
        )
        if not torch.isfinite(step_dir).all():
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio)

        policy_loss = -self._safe_float(loss)
        loss = loss.detach().cpu().numpy()

        params = flat_params(self.actor)
        fvp = fisher_vector_product(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            step_dir,
        )
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        if (not torch.isfinite(shs).all()) or shs.item() <= 1e-12:
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir
        if not torch.isfinite(full_step).all():
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio)

        old_actor = StochasticPolicy(
            self.args, self.obs_space, self.act_space, self.device
        )
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        if (not torch.isfinite(expected_improve).all()) or expected_improve.item() <= 1e-12:
            update_model(self.actor, params)
            return self._skipped_update(dist_entropy=dist_entropy, ratio=ratio)
        expected_improve = expected_improve.detach().cpu().numpy()

        # Backtracking line search (https://en.wikipedia.org/wiki/Backtracking_line_search)
        flag = False
        fraction = 1
        kl = torch.tensor(0.0, device=self.device)
        loss_improve = np.array([0.0], dtype=np.float32)
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            if not torch.isfinite(new_params).all():
                expected_improve *= self.backtrack_coeff
                fraction *= self.backtrack_coeff
                continue
            update_model(self.actor, new_params)
            action_log_probs, dist_entropy, _ = self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )
            if not (
                torch.isfinite(action_log_probs).all()
                and torch.isfinite(dist_entropy).all()
            ):
                update_model(self.actor, params)
                expected_improve *= self.backtrack_coeff
                fraction *= self.backtrack_coeff
                continue

            log_ratio = torch.clamp(
                action_log_probs - old_action_log_probs_batch,
                min=-20.0,
                max=20.0,
            )
            ratio = getattr(torch, self.action_aggregation)(
                torch.exp(log_ratio),
                dim=-1,
                keepdim=True,
            )
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=20.0, neginf=0.05)
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / denom
            else:
                new_loss = torch.sum(
                    ratio * factor_batch * adv_targ, dim=-1, keepdim=True
                ).mean()
            if not torch.isfinite(new_loss):
                update_model(self.actor, params)
                expected_improve *= self.backtrack_coeff
                fraction *= self.backtrack_coeff
                continue

            new_loss = new_loss.detach().cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                new_actor=self.actor,
                old_actor=old_actor,
            )
            kl = kl.mean()
            if not torch.isfinite(kl):
                update_model(self.actor, params)
                expected_improve *= self.backtrack_coeff
                fraction *= self.backtrack_coeff
                continue
            improve_ratio = loss_improve / expected_improve

            if (
                kl < self.kl_threshold
                and np.isfinite(improve_ratio).all()
                and improve_ratio > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.actor, params)
            print("policy update does not impove the surrogate")

        return {
            "kl": self._safe_float(kl),
            "policy_loss": policy_loss,
            "loss_improve": self._safe_float(loss_improve),
            "expected_improve": self._safe_float(expected_improve),
            "dist_entropy": self._safe_float(dist_entropy),
            "ratio": self._safe_float(ratio, default=1.0),
            "actor_grad_norm": self._safe_float(actor_grad_norm),
            "trpo_step_norm": self._safe_float(torch.norm(full_step)),
            "trpo_update_accepted": 1.0 if flag else 0.0,
            "trpo_skipped_nonfinite": 0.0,
            "trpo_skipped_no_signal": 0.0,
        }

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0
        train_info["policy_loss"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["trpo_step_norm"] = 0
        train_info["trpo_update_accepted"] = 0
        train_info["trpo_skipped_nonfinite"] = 0
        train_info["trpo_skipped_no_signal"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        advantages = np.nan_to_num(
            advantages.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            if (not np.isfinite(mean_advantages)) or (not np.isfinite(std_advantages)) or std_advantages < 1e-8:
                advantages = np.zeros_like(advantages, dtype=np.float32)
            else:
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        advantages = np.clip(advantages, -10.0, 10.0).astype(np.float32, copy=False)

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        num_updates = 0
        for sample in data_generator:
            update_info = self.update(sample)
            for k in train_info.keys():
                train_info[k] += self._safe_float(update_info.get(k, 0.0))
            num_updates += 1
        num_updates = max(num_updates, 1)

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
