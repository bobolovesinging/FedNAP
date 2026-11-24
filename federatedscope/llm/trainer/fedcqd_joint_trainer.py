"""Joint Adversarial Head Trainer — Single MLP with multi-factor outputs.

This module provides a *single* adversarial head whose shared MLP body
feeds multiple output logits (one per bias factor).  All factors share
the same hidden representation, and a single GRL reverses the gradient
once — producing a balanced de-biasing signal across all factors.

Architecture
------------
    z ──→ [GRL] ──→ [shared MLP body] ──→ [output logit₀, output logit₁]
                                              ↓               ↓
                                          sigmoid(p₀)     sigmoid(p₁)
                                          (length)         (lead)

Contrast with the default GRLAdversarialSelector which uses *separate*
MLP heads, each with its own GRL.

Usage
-----
In YAML, set:

    trainer:
      type: fedcqd_trainer
      causal_selector_type: joint_grl_adversarial
      adv_factors:
        - bias_length
        - bias_lead

No changes to ``fedcqd_trainer.py`` are required — this module auto-
registers itself when imported (the trainer ``__init__.py`` globs all
``*.py`` files in this directory).
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from federatedscope.llm.trainer.fedcqd_trainer import (
    GRLAdversarialSelector,
    GRL,
    MLP,
    SELECTOR_REGISTRY,
    logger,
)

logger = logging.getLogger(__name__)


# =====================================================================
# JointAdversarialHead — one MLP body, multiple output logits
# =====================================================================

class JointAdversarialHead(nn.Module):
    """Single MLP with shared body + multi-output.

    Args:
        input_size: Dimension of input features.
        num_factors: Number of bias factors (output logits).
        hidden_size: Size of shared hidden layer.
        num_hidden: Number of shared hidden layers (body depth).
        batch_norm: Whether to use batch normalization in the body.
        dropout: Dropout rate in the body.
    """

    def __init__(self, input_size, num_factors, hidden_size=256,
                 num_hidden=1, batch_norm=False, dropout=0.1):
        super().__init__()
        self.num_factors = num_factors
        self.grl = GRL()

        self.body = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            output_size=hidden_size,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_size, num_factors)

    def forward(self, z, lambda_=1.0, use_grl=True):
        if use_grl:
            self.grl.lambda_ = lambda_
            z = self.grl(z)
        h = self.body(z)                    # [B, hidden_size]
        logits = self.output_layer(h)       # [B, num_factors]
        return torch.sigmoid(logits)        # [B, num_factors]


# =====================================================================
# _FactorSlice — thin wrapper that extracts one factor from joint head
# =====================================================================

class _FactorSlice:
    """Callable that delegates to joint_head and slices one factor.

    Used as a drop-in replacement for ``self.adv_heads[i]`` in code
    paths (e.g. ``_compute_selector_direct_loss``) that expect a
    ModuleList of per-factor heads.  Not a nn.Module — does NOT
    contribute to state_dict, keeping the checkpoint compatible with
    strict=True loading.
    """
    def __init__(self, joint_head, factor_idx):
        self._joint_head = joint_head
        self._idx = factor_idx

    def __call__(self, z, lambda_=1.0, use_grl=True):
        pred_all = self._joint_head(z, lambda_=lambda_, use_grl=use_grl)
        return pred_all[:, self._idx:self._idx + 1]


# =====================================================================
# JointGRLAdversarialSelector — uses single JointAdversarialHead
# =====================================================================

class JointGRLAdversarialSelector(GRLAdversarialSelector):
    """Selector with a single joint adversarial head.

    Inherits all configuration, warmup, early-stopping, and checkpoint
    logic from GRLAdversarialSelector.  Only overrides:
      - ``__init__``: create JointAdversarialHead + compatible adv_heads
      - ``compute_loss``: single forward, multi-output BCE
      - ``evaluate``: single forward, multi-output accuracy
    """

    def __init__(self, hidden_size, adv_factors, trainer_config):
        # Parent sets up adv_heads as ModuleList (will be kept but unused).
        super().__init__(hidden_size, adv_factors, trainer_config)

        num_factors = len(self.adv_factors)
        self.joint_head = JointAdversarialHead(
            input_size=self.input_size,
            num_factors=num_factors,
            hidden_size=hidden_size,
            num_hidden=1,
            batch_norm=True,
            dropout=self.adv_dropout,
        )

        # Replace parent's adv_heads with empty ModuleList so that
        # the state_dict only contains joint_head keys, matching the
        # Phase 1 checkpoint format for strict=True loading.
        # _compute_selector_direct_loss is monkey-patched to handle
        # JointGRLAdversarialSelector via joint_head directly.
        self.adv_heads = nn.ModuleList([])
        self._num_factors = num_factors

        logger.info(
            "[JointGRLSelector] input_size=%d num_factors=%d "
            "hidden_size=%d (shared body + %d outputs)",
            self.input_size, num_factors, hidden_size, num_factors,
        )

    def compute_loss(self, win_h, lose_h, data_batch, use_grl, device,
                     factor_lambdas=None):
        adv_loss = torch.tensor(0.0, device=device)
        metrics = {}

        if not self.adv_factors:
            return adv_loss, metrics

        z = self._prepare_input(win_h, lose_h)

        if factor_lambdas is not None:
            effective_lambda = (
                sum(factor_lambdas.values()) / len(factor_lambdas)
                if factor_lambdas else 0.0
            )
        else:
            effective_lambda = self.grl_lambda

        pred_all = self.joint_head(
            z, lambda_=effective_lambda, use_grl=use_grl
        )  # [B, num_factors]

        for i, factor in enumerate(self.adv_factors):
            if factor not in data_batch:
                continue

            lbl = data_batch[factor].float().to(device)
            if lbl.dim() == 1:
                lbl = lbl.unsqueeze(1)

            smooth_lbl = self._smooth_labels(lbl, self.adv_label_smoothing)
            pred_i = pred_all[:, i:i + 1]

            this_loss = F.binary_cross_entropy(pred_i, smooth_lbl)
            adv_loss = adv_loss + this_loss

            with torch.no_grad():
                pred_bin = (pred_i >= 0.5).long()
                correct = (pred_bin == lbl.long()).sum().item()
                total = lbl.numel()
                metrics[f'adv_{factor}_acc'] = (
                    correct / total if total > 0 else 0.0)
                metrics[f'adv_{factor}_loss'] = this_loss.item()

            if not use_grl and not hasattr(self, '_adv_pred_diag_done'):
                with torch.no_grad():
                    logger.info(
                        '[JointGRLSelector][diag] factor=%s '
                        'pred_min=%.4f pred_max=%.4f pred_mean=%.4f '
                        'pred_std=%.4f (B=%d)',
                        factor, pred_i.min().item(),
                        pred_i.max().item(), pred_i.mean().item(),
                        pred_i.std().item(), total)

        if not use_grl:
            self._adv_pred_diag_done = True

        metrics['adv_loss'] = adv_loss.item()
        return adv_loss, metrics

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, test_batches, encode_fn, device):
        if not test_batches or not self.adv_factors:
            return {}

        factor_correct = {f: 0 for f in self.adv_factors}
        factor_total = {f: 0 for f in self.adv_factors}

        self.eval()
        with torch.no_grad():
            for batch in test_batches:
                win_h, lose_h = encode_fn(batch)
                z = self._prepare_input(win_h, lose_h)

                pred_all = self.joint_head(
                    z, lambda_=1.0, use_grl=False
                )

                for i, factor in enumerate(self.adv_factors):
                    if factor not in batch:
                        continue
                    lbl = batch[factor].float().to(device)
                    if lbl.dim() == 1:
                        lbl = lbl.unsqueeze(1)
                    pred_i = pred_all[:, i:i + 1]
                    pred_bin = (pred_i >= 0.5).long()
                    factor_correct[factor] += (
                        (pred_bin == lbl.long()).sum().item())
                    factor_total[factor] += lbl.numel()

        self.train()

        total_correct = sum(factor_correct.values())
        total_samples = sum(factor_total.values())
        test_adv_acc = (
            total_correct / total_samples if total_samples > 0 else 0.0)

        per_factor = {
            f: factor_correct[f] / factor_total[f]
            for f in self.adv_factors if factor_total[f] > 0
        }

        if test_adv_acc > self._best_adv_acc:
            self._best_adv_acc = test_adv_acc
            self._no_improve_rounds = 0
        else:
            self._no_improve_rounds += 1

        return {
            'test_adv_acc': test_adv_acc,
            'per_factor': per_factor,
        }


# =====================================================================
# Monkey-patch FedCQDTrainer._compute_selector_direct_loss for joint
# =====================================================================
# When causal_selector_type == 'joint_grl_adversarial', the
# _compute_selector_direct_loss method in FedCQDTrainer calls
#    self.selector.adv_heads[i](z, lambda_, use_grl=False)
# but adv_heads is a plain ModuleList we can't replace with a property.
# We patch it at module load time once FedCQDTrainer is available.

import federatedscope.llm.trainer.fedcqd_trainer as _fedcqd_module

_original_compute_selector_direct = (
    _fedcqd_module.FedCQDTrainer._compute_selector_direct_loss)


def _patched_compute_selector_direct_loss(
        self, z_detached, data_batch, device):
    """Patched version that handles joint_grl_adversarial selector."""
    selector = self.selector
    if isinstance(selector, JointGRLAdversarialSelector):
        adv_loss = torch.tensor(0.0, device=device)
        pred_all = selector.joint_head(
            z_detached, lambda_=1.0, use_grl=False)
        for i, factor in enumerate(selector.adv_factors):
            if factor not in data_batch:
                continue
            lbl = data_batch[factor].float().to(device)
            if lbl.dim() == 1:
                lbl = lbl.unsqueeze(1)
            smooth_lbl = selector._smooth_labels(
                lbl, selector.adv_label_smoothing)
            pred_i = pred_all[:, i:i + 1]
            this_loss = F.binary_cross_entropy(pred_i, smooth_lbl)
            adv_loss = adv_loss + this_loss
        return adv_loss
    else:
        # Original logic for non-joint selectors
        adv_loss = torch.tensor(0.0, device=device)
        for i, factor in enumerate(selector.adv_factors):
            if factor not in data_batch:
                continue
            lbl = data_batch[factor].float().to(device)
            if lbl.dim() == 1:
                lbl = lbl.unsqueeze(1)
            smooth_lbl = selector._smooth_labels(
                lbl, selector.adv_label_smoothing)
            pred = selector.adv_heads[i](
                z_detached, lambda_=1.0, use_grl=False)
            this_loss = F.binary_cross_entropy(pred, smooth_lbl)
            adv_loss = adv_loss + this_loss
        return adv_loss


# Replace the instance method — we can't easily monkey-patch a bound method,
# so we patch at the class level by overriding the internal call in
# _hook_on_batch_forward.  The actual call site is:
#   adv_loss_direct = self._compute_selector_direct_loss(
#       self.selector, win_h_2, lose_h_2, dev, ctx.data_batch)
# So we replace the class method.
_fedcqd_module.FedCQDTrainer._compute_selector_direct_loss = (
    _patched_compute_selector_direct_loss)


# =====================================================================
# Auto-register in SELECTOR_REGISTRY
# =====================================================================

SELECTOR_REGISTRY['joint_grl_adversarial'] = JointGRLAdversarialSelector

logger.info(
    "[JointGRLTrainer] Registered 'joint_grl_adversarial' selector. "
    "Use causal_selector_type: joint_grl_adversarial in YAML."
)
