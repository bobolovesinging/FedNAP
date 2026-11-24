"""FedCQD Trainer - Federated Causal + Quality-Diversity

Extends FedBiscuit's RewardChoiceTrainer with:
1. GRL-based adversarial training for causal disentanglement
2. Two-phase training: Phase1 pretrain adv heads, Phase2 joint training
3. Extensible CausalSelector interface for future methods

Improvements over predecessor's implementation:
- Concat input representation [win_h, lose_h, z] for richer bias signal
- Independent learning rate for adversarial heads
- Label smoothing to prevent gradient saturation
- Dropout in adversarial heads for regularization
- Progressive lambda warmup in Phase2
- Phase1 early stopping based on held-out adv accuracy

Ablation experiment guide (YAML trainer section):
  # To revert to predecessor's behavior (for ablation baseline):
  adv_input_mode: diff           # original: only z = win_h - lose_h
  adv_lr: 5.0e-05               # same as main lr (no independent lr)
  adv_label_smoothing: 0.0      # no smoothing
  adv_dropout: 0.0              # no dropout
  adv_warmup_rounds: 0          # no warmup, full lambda from start
  adv_early_stop_patience: 0    # no early stopping
  causal_selector_type: grl_adversarial
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import random
from abc import ABC, abstractmethod

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)


def build_independent_optimizer(base_optimizer,
                                parameters,
                                lr,
                                weight_decay):
    """Build an optimizer from the implementation under Accelerate."""
    unwrapped = getattr(base_optimizer, 'optimizer', base_optimizer)
    optimizer_cls = type(unwrapped)
    kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    if 'betas' in unwrapped.defaults:
        kwargs['betas'] = unwrapped.defaults['betas']
    if 'eps' in unwrapped.defaults:
        kwargs['eps'] = unwrapped.defaults['eps']
    return optimizer_cls(parameters, **kwargs)

# =====================================================================
# Constants
# =====================================================================

ADV_TRAIN_SIZE = 8000  # first N samples for phase-1 training (was 90)
ADV_TEST_SIZE = 1000   # held-out samples for adv evaluation (was 10)


# =====================================================================
# Gradient Reversal Layer
# =====================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_
        return grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


# =====================================================================
# MLP with optional dropout
# =====================================================================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden, output_size,
                 activation=None, batch_norm=False, dropout=0.0):
        super().__init__()
        if activation is None:
            activation = nn.GELU()

        self.layers = nn.ModuleList()

        if num_hidden <= 0:
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            cur_in = input_size
            for _ in range(num_hidden):
                self.layers.append(nn.Linear(cur_in, hidden_size))
                if batch_norm:
                    self.layers.append(nn.LayerNorm(hidden_size))
                self.layers.append(activation)
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
                cur_in = hidden_size
            self.layers.append(nn.Linear(cur_in, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =====================================================================
# AdversarialHead (enhanced with configurable input size and dropout)
# =====================================================================

class AdversarialHead(nn.Module):
    """MLP + sigmoid adversarial classifier with optional GRL.

    Args:
        input_size: Dimension of input features (may differ from hidden_size
                    if using concat input mode: 3*hidden_size)
        hidden_size: Size of hidden layers
        num_hidden: Number of hidden layers
        batch_norm: Whether to use batch normalization
        dropout: Dropout rate (0.0 = no dropout)
    """
    def __init__(self, input_size, hidden_size=256, num_hidden=1,
                 batch_norm=False, dropout=0.1):
        super().__init__()
        self.grl = GRL()
        self.decoder = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden=num_hidden,
            output_size=1,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, z, lambda_=1.0, use_grl=True):
        if use_grl:
            self.grl.lambda_ = lambda_
            z = self.grl(z)
        return torch.sigmoid(self.decoder(z))


# =====================================================================
# Helper functions
# =====================================================================

def _infer_hidden_size(model, fallback=1024):
    for path in ('config', 'model.config', 'base_model.config',
                 'base_model.model.config'):
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if hasattr(obj, 'hidden_size'):
                return int(obj.hidden_size)
        except AttributeError:
            continue
    return fallback


def _last_real_token(hidden, mask):
    """hidden:[B,T,H] mask:[B,T] -> [B,H]"""
    idx = (mask.sum(1) - 1).clamp(min=0)
    bsz = hidden.size(0)
    return hidden[torch.arange(bsz, device=hidden.device), idx, :].float()


def pool_adversarial_hidden(hidden, mask, labels, mode='last'):
    if mode == 'last':
        return _last_real_token(hidden, mask)
    if mode != 'response_mean':
        raise ValueError(f'Unknown adversarial pooling mode: {mode}')
    if labels is None or labels.shape != mask.shape:
        raise ValueError('Response pooling requires aligned response labels')
    response = (labels.to(mask.device) != -100) & mask.bool()
    # Complete pair encodings append EOS; it is not part of the response text.
    last = (mask.sum(1) - 1).clamp_min(0)
    if not response[torch.arange(len(mask), device=mask.device), last].all():
        raise ValueError('Response pooling found a missing/truncated response')
    response = response.clone()
    response[torch.arange(len(mask), device=mask.device), last] = False
    count = response.sum(1, keepdim=True)
    # An audited, genuinely empty answer has only EOS and maps to zero.
    return (hidden.float() * response.unsqueeze(-1)).sum(1) / count.clamp_min(1)


# =====================================================================
# CausalSelectorBase - Abstract interface for extensibility
# =====================================================================

class CausalSelectorBase(nn.Module, ABC):
    """Abstract base class for causal disentanglement selectors.

    A selector encapsulates the causal disentanglement logic:
    - How to compute the disentanglement loss
    - How to evaluate disentanglement quality

    Subclasses implement specific methods (GRL+adversarial,
    counterfactual regularization, HSIC, etc.).

    To add a new method:
    1. Create a subclass of CausalSelectorBase
    2. Implement compute_loss() and evaluate()
    3. Register it in SELECTOR_REGISTRY below
    4. Set trainer.causal_selector_type in YAML config
    """

    @abstractmethod
    def compute_loss(self, win_h, lose_h, data_batch, use_grl, device):
        """Compute the causal disentanglement loss.

        Args:
            win_h: Winner hidden states [B, H]
            lose_h: Loser hidden states [B, H]
            data_batch: Dict containing bias factor labels
            use_grl: Whether to apply gradient reversal
                     (Phase1=False, Phase2=True)
            device: torch device

        Returns:
            (loss_tensor, metrics_dict)
            metrics_dict: per-factor accuracy and loss for logging
        """
        pass

    @abstractmethod
    def evaluate(self, test_batches, encode_fn, device):
        """Evaluate disentanglement quality on held-out test data.

        Args:
            test_batches: List of held-out batch dicts
            encode_fn: Callable(batch) -> (win_h, lose_h) with no_grad
            device: torch device

        Returns:
            dict with 'test_adv_acc', 'per_factor', etc.
        """
        pass

    def get_effective_lambda(self, phase2_round_idx):
        """Get effective lambda_adv (supports progressive warmup).

        Override in subclasses that support warmup.
        Returns 0.0 by default (no lambda contribution).
        """
        return 0.0

    def should_early_stop(self, patience):
        """Check if training should early stop.

        Override in subclasses that support early stopping.
        Returns False by default.
        """
        return False

    def on_phase2_start(self):
        """Called when transitioning from Phase1 to Phase2."""
        pass

    def advance_phase2_counter(self):
        """Increment Phase2 round counter. Override if warmup is used."""
        pass


# =====================================================================
# GRLAdversarialSelector - Current implementation (improved)
# =====================================================================

class GRLAdversarialSelector(CausalSelectorBase):
    """GRL + Adversarial Head selector with improved training stability.

    Key improvements over predecessor:
    1. Concat input [win_h, lose_h, z] for richer bias signal
       (configurable via adv_input_mode)
    2. Label smoothing to prevent gradient saturation in sigmoid
    3. Dropout in adversarial heads for regularization
    4. Independent learning rate for adversarial heads
    5. Progressive lambda warmup in Phase2
    6. Phase1 early stopping based on held-out accuracy

    YAML configuration (all have sensible defaults):
        trainer:
          causal_selector_type: grl_adversarial  # selector type
          adv_input_mode: concat    # "concat"=[win,lose,z] or "diff"=z only
          adv_lr: 0.001             # independent lr for adv heads
          adv_label_smoothing: 0.1  # BCE label smoothing
          adv_dropout: 0.1          # dropout in adv heads
          adv_warmup_rounds: 3      # Phase2 lambda warmup rounds
          adv_early_stop_patience: 2  # Phase1 early stop patience
    """

    def __init__(self, hidden_size, adv_factors, trainer_config):
        super().__init__()
        self.hidden_size = hidden_size
        self.adv_factors = list(adv_factors) if adv_factors else []

        # ── Config with improved defaults ──────────────────────────────
        # grl_lambda: supports both scalar (all heads share same λ) and
        # dict (per-factor λ, e.g. {'bias_length': 0.2, 'bias_lead': 0.5}).
        raw_grl_lambda = getattr(trainer_config, 'grl_lambda', 1.0)
        if isinstance(raw_grl_lambda, dict):
            self.grl_lambdas = {
                f: float(raw_grl_lambda.get(f, 1.0))
                for f in self.adv_factors
            }
        elif isinstance(raw_grl_lambda, (int, float)):
            self.grl_lambdas = {
                f: float(raw_grl_lambda) for f in self.adv_factors
            }
        else:
            self.grl_lambdas = {f: 1.0 for f in self.adv_factors}
        # Legacy attribute: max lambda across all heads (used for > 0 checks)
        self.grl_lambda = (
            max(self.grl_lambdas.values()) if self.grl_lambdas else 0.0)
        self.lambda_adv = float(getattr(trainer_config, 'lambda_adv', 1.0))
        self.adv_input_mode = getattr(trainer_config, 'adv_input_mode', 'concat')
        self.adv_lr = float(getattr(trainer_config, 'adv_lr', 1e-3))
        self.adv_label_smoothing = float(
            getattr(trainer_config, 'adv_label_smoothing', 0.1))
        self.adv_dropout = float(getattr(trainer_config, 'adv_dropout', 0.1))
        head_width = getattr(trainer_config, 'adv_head_hidden_size', hidden_size)
        head_layers = getattr(trainer_config, 'adv_head_num_hidden', 1)
        if (type(head_width) is not int or head_width <= 0
                or type(head_layers) is not int or head_layers < 0):
            raise ValueError('Adversarial head width/layers must be valid integers')
        self.head_architecture = {'hidden_size': head_width,
                                  'num_hidden': head_layers}
        self.adv_warmup_rounds = int(
            getattr(trainer_config, 'adv_warmup_rounds', 3))
        self.adv_early_stop_patience = int(
            getattr(trainer_config, 'adv_early_stop_patience', 2))

        # ── Input size based on mode ──────────────────────────────────
        if self.adv_input_mode == 'concat':
            self.input_size = hidden_size * 3  # [win_h, lose_h, z]
        else:
            self.input_size = hidden_size       # z = win_h - lose_h (original)

        # ── Adversarial heads ─────────────────────────────────────────
        self.adv_heads = nn.ModuleList([
            AdversarialHead(
                input_size=self.input_size,
                hidden_size=head_width,
                num_hidden=head_layers,
                batch_norm=True,
                dropout=self.adv_dropout,
            )
            for _ in self.adv_factors
        ])

        # ── Warmup state ──────────────────────────────────────────────
        self._phase2_round_counter = 0

        # ── Early stopping state ──────────────────────────────────────
        self._best_adv_acc = 0.0
        self._no_improve_rounds = 0

        logger.info(
            '[GRLAdversarialSelector] head_width=%d head_layers=%d parameters=%d',
            head_width, head_layers,
            sum(parameter.numel() for parameter in self.adv_heads.parameters()))
        logger.info(
            "[GRLAdversarialSelector] input_mode=%s input_size=%d "
            "grl_lambdas=%s grl_lambda_max=%.2f lambda_adv=%.2f adv_lr=%.1e "
            "label_smoothing=%.2f dropout=%.2f warmup_rounds=%d "
            "early_stop_patience=%d factors=%s",
            self.adv_input_mode, self.input_size,
            self.grl_lambdas, self.grl_lambda, self.lambda_adv, self.adv_lr,
            self.adv_label_smoothing, self.adv_dropout,
            self.adv_warmup_rounds, self.adv_early_stop_patience,
            self.adv_factors
        )

    # ── Input preparation ─────────────────────────────────────────────

    def _prepare_input(self, win_h, lose_h, factor=None):
        """Prepare input for adversarial heads based on input_mode.

        'concat': [win_h, lose_h, win_h - lose_h]  (3 * hidden_size)
                  Preserves directionality and per-response features
        'diff':   win_h - lose_h                    (hidden_size)
                  Original predecessor approach
        """
        if isinstance(win_h, dict) or isinstance(lose_h, dict):
            if (not isinstance(win_h, dict) or not isinstance(lose_h, dict)
                    or factor not in win_h or factor not in lose_h):
                raise ValueError('Mixed head representations require an explicit factor')
            win_h, lose_h = win_h[factor], lose_h[factor]
        z = win_h - lose_h
        if self.adv_input_mode == 'concat':
            return torch.cat([win_h, lose_h, z], dim=-1)
        return z

    # ── Label smoothing ───────────────────────────────────────────────

    @staticmethod
    def _smooth_labels(labels, smoothing):
        """Apply label smoothing: hard 0/1 -> soft labels.

        Prevents sigmoid output saturation and gradient vanishing:
            0 -> smoothing * 0.5  (e.g., 0.05)
            1 -> 1 - smoothing * 0.5  (e.g., 0.95)
        """
        if smoothing <= 0:
            return labels
        return labels * (1.0 - smoothing) + 0.5 * smoothing

    # ── Core loss computation ─────────────────────────────────────────

    def compute_loss(self, win_h, lose_h, data_batch, use_grl, device,
                     factor_lambdas=None):
        """Compute adversarial loss for all bias factors.

        Args:
            win_h: [B, H] winner hidden states
            lose_h: [B, H] loser hidden states
            data_batch: dict with bias factor labels
            use_grl: True for Phase2 (gradient reversal), False for Phase1
            device: torch device
            factor_lambdas: optional dict {factor: lambda} to override
                self.grl_lambdas per factor. Used by Phase 2 Forward Pass 2
                to pass warmup-scaled effective lambdas.

        Returns:
            (loss_tensor, metrics_dict)
        """
        adv_loss = torch.tensor(0.0, device=device)
        metrics = {}

        if not self.adv_factors:
            return adv_loss, metrics

        for i, factor in enumerate(self.adv_factors):
            if factor not in data_batch:
                continue
            z = self._prepare_input(win_h, lose_h, factor=factor)

            lbl = data_batch[factor].float().to(device)
            if lbl.dim() == 1:
                lbl = lbl.unsqueeze(1)

            # Apply label smoothing for training stability
            smooth_lbl = self._smooth_labels(lbl, self.adv_label_smoothing)

            # Per-factor lambda: use override if provided, else default
            factor_lambda = (
                factor_lambdas.get(factor, self.grl_lambdas.get(factor, 0.0))
                if factor_lambdas is not None
                else self.grl_lambdas.get(factor, 0.0)
            )

            pred = self.adv_heads[i](
                z, lambda_=factor_lambda, use_grl=use_grl)
            this_loss = F.binary_cross_entropy(pred, smooth_lbl)
            adv_loss = adv_loss + this_loss

            # Collect per-factor metrics for logging
            with torch.no_grad():
                pred_bin = (pred >= 0.5).long()
                correct = (pred_bin == lbl.long()).sum().item()
                total = lbl.numel()
                metrics[f'adv_{factor}_acc'] = (
                    correct / total if total > 0 else 0.0)
                metrics[f'adv_{factor}_loss'] = this_loss.item()

            # Diagnostic: check if head predictions are "dead" (constant)
            if not use_grl and not hasattr(self, '_adv_pred_diag_done'):
                with torch.no_grad():
                    pred_min = pred.min().item()
                    pred_max = pred.max().item()
                    pred_mean = pred.mean().item()
                    pred_std = pred.std().item()
                    import logging
                    _logger = logging.getLogger(__name__.split('.')[0])
                    _logger.info(
                        '[GRLSelector][diag] factor=%s pred_min=%.4f '
                        'pred_max=%.4f pred_mean=%.4f pred_std=%.4f '
                        'pred_const=%s (B=%d)',
                        factor, pred_min, pred_max, pred_mean, pred_std,
                        pred_min == pred_max, total)

        metrics['adv_loss'] = adv_loss.item()
        return adv_loss, metrics

    # ── Progressive lambda warmup ─────────────────────────────────────

    def get_effective_lambda(self, phase2_round_idx):
        """Get effective lambda_adv with progressive warmup.

        lambda_eff = lambda_adv * min(1.0, (round + 1) / warmup_rounds)

        With warmup_rounds=3:
          Phase2 round 0: lambda_eff = lambda_adv * 1/3
          Phase2 round 1: lambda_eff = lambda_adv * 2/3
          Phase2 round 2: lambda_eff = lambda_adv * 3/3 = lambda_adv
          Phase2 round 3+: lambda_eff = lambda_adv (full)
        """
        if self.lambda_adv <= 0:
            return self.lambda_adv
        if self.adv_warmup_rounds <= 0:
            return self.lambda_adv
        warmup_factor = min(
            1.0, (phase2_round_idx + 1) / self.adv_warmup_rounds)
        return self.lambda_adv * warmup_factor

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, test_batches, encode_fn, device):
        """Evaluate adversarial heads on held-out test split.

        Args:
            test_batches: list of batch dicts
            encode_fn: callable(batch) -> (win_h, lose_h)
            device: torch device

        Returns:
            dict with 'test_adv_acc', 'per_factor'
        """
        if not test_batches or not self.adv_factors:
            return {}

        factor_correct = {f: 0 for f in self.adv_factors}
        factor_total = {f: 0 for f in self.adv_factors}

        self.eval()
        with torch.no_grad():
            for batch in test_batches:
                win_h, lose_h = encode_fn(batch)
                for i, factor in enumerate(self.adv_factors):
                    if factor not in batch:
                        continue
                    z = self._prepare_input(win_h, lose_h, factor=factor)
                    lbl = batch[factor].float().to(device)
                    if lbl.dim() == 1:
                        lbl = lbl.unsqueeze(1)
                    pred = self.adv_heads[i](
                        z, lambda_=1.0, use_grl=False)
                    pred_bin = (pred >= 0.5).long()
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

        # Update early stopping state
        if test_adv_acc > self._best_adv_acc:
            self._best_adv_acc = test_adv_acc
            self._no_improve_rounds = 0
        else:
            self._no_improve_rounds += 1

        return {
            'test_adv_acc': test_adv_acc,
            'per_factor': per_factor,
        }

    # ── Early stopping ────────────────────────────────────────────────

    def should_early_stop(self, patience):
        """Check if Phase1 should early stop.

        Args:
            patience: Number of eval rounds without improvement.
                      0 = disabled (never early stop).
        """
        return patience > 0 and self._no_improve_rounds >= patience

    # ── Phase transition hooks ────────────────────────────────────────

    def on_phase2_start(self):
        """Reset Phase2 warmup counter when transitioning from Phase1."""
        self._phase2_round_counter = 0

    def advance_phase2_counter(self):
        """Increment Phase2 round counter and return new value."""
        self._phase2_round_counter += 1
        return self._phase2_round_counter


# =====================================================================
# Selector Registry
# =====================================================================

SELECTOR_REGISTRY = {
    'grl_adversarial': GRLAdversarialSelector,
}


def create_selector(selector_type, hidden_size, adv_factors, trainer_config):
    """Factory function to create a causal selector by type name."""
    if selector_type not in SELECTOR_REGISTRY:
        raise ValueError(
            f"[FedCQD] Unknown causal_selector_type='{selector_type}'. "
            f"Available: {list(SELECTOR_REGISTRY.keys())}"
        )
    cls = SELECTOR_REGISTRY[selector_type]
    return cls(hidden_size, adv_factors, trainer_config)


# =====================================================================
# FedCQDTrainer
# =====================================================================

class FedCQDTrainer(LLMTrainer):
    """FedCQD Trainer with pluggable causal disentanglement selector.

    Two-phase training:
      Phase 1 (rounds 0..adv_pretrain_rounds-1):
        - Freeze LLM + score_head
        - Train adversarial selector only on labeled bias data
      Phase 2 (rounds adv_pretrain_rounds..total_round_num-1):
        - Train score_head + selected LoRA layers
        - Adversarial selector uses GRL for gradient reversal
        - total_loss = reward_loss - effective_lambda * adv_loss

    The selector is pluggable via causal_selector_type:
      - 'grl_adversarial': GRL + adversarial heads (current)
      - Future: 'counterfactual', 'hsic', etc.
    """

    def __init__(self, model, data, device, config,
                 only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        # ── Basic config ──────────────────────────────────────────────
        raw = getattr(config.trainer, 'adv_factors', [])
        self.adv_factors = list(raw) if raw else []

        fallback = int(getattr(config.trainer, 'hidden_size', 1024))
        self.hidden_size = _infer_hidden_size(model, fallback)

        # Freeze LLM ablation: when True, skip LoRA in Phase 2 so only
        # score_head is trained (LLM encoder stays completely frozen).
        # Useful for ablation: "frozen LLM + trainable score_head" vs
        # "LoRA-tuned LLM + trainable score_head".
        self.freeze_llm = getattr(config.trainer, 'freeze_llm', False)

        # Min-max mode: when True, Phase 2 trains ALL components
        # simultaneously (LLM + score_head + selector). The selector is
        # NOT frozen — it plays the "max" role in the min-max game.
        # Step 3 backward (adv_loss_direct → selector) is enabled.
        self.minmax_mode = getattr(config.trainer, 'minmax_mode', False)

        # Backward compat: stage1_steps is an alias for adv_pretrain_rounds
        self.adv_pretrain_rounds = int(
            getattr(config.trainer, 'adv_pretrain_rounds',
                    getattr(config.trainer, 'stage1_steps', 3))
        )

        # Pretrained adversarial head path (skip Phase 1 if provided)
        self.adv_pretrained_path = getattr(
            config.trainer, 'adv_pretrained_path', '')

        # Per-factor pretrained head paths (new: for dual-head experiments).
        # Dict of {factor_name: path_to_adv_pretrained.pt}, e.g.
        #   adv_pretrained_heads:
        #     bias_length: exp/tldr/phase1_length/adv_pretrained.pt
        #     bias_lead: exp/tldr/phase1_lead/adv_pretrained.pt
        # Each checkpoint contains a selector trained on a single factor;
        # we extract only that factor's head weights and load them into
        # the corresponding slot in self.selector.adv_heads[i].
        raw_heads = getattr(config.trainer, 'adv_pretrained_heads', None)
        if raw_heads:
            # Filter out yacs internal keys (__cfg_check_funcs__, etc.)
            self.adv_pretrained_heads = {
                k: v for k, v in raw_heads.items()
                if not k.startswith('_') and k != 'is_ready_for_run'
            }
        else:
            self.adv_pretrained_heads = {}

        # Selector type
        self.causal_selector_type = getattr(
            config.trainer, 'causal_selector_type', 'grl_adversarial')

        # ── Create causal selector ────────────────────────────────────
        self.selector = create_selector(
            self.causal_selector_type,
            self.hidden_size,
            self.adv_factors,
            config.trainer,
        ).to(device)

        # ── Score head (reward model) ─────────────────────────────────
        # Xavier initialization for stable reward scores (prevents NaN in
        # logsigmoid when score_head output is too large initially)
        self.score_head = nn.Linear(self.hidden_size, 1, bias=True).to(device)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

        # ── Load pretrained adversarial head (if provided) ────────────
        # When adv_pretrained_path is set, load selector weights from a
        # prior Phase 1 run and skip Phase 1 entirely.
        # Note: score_head is NOT loaded — Phase 1 doesn't train it
        # (it's frozen during Phase 1), so its saved weights are just
        # random init.  Phase 2 will train score_head from scratch.
        self._adv_loaded_from_disk = False
        if self.adv_pretrained_path and os.path.isfile(self.adv_pretrained_path):
            ckpt = torch.load(self.adv_pretrained_path, map_location=device)
            if 'selector' in ckpt:
                self.selector.load_state_dict(ckpt['selector'])
                logger.info(
                    "[FedCQD] Loaded pretrained selector from %s "
                    "(adv_factors=%s, trained_rounds=%d)",
                    self.adv_pretrained_path,
                    ckpt.get('adv_factors', '?'),
                    ckpt.get('trained_rounds', '?'),
                )
            self._adv_loaded_from_disk = True
            self._phase1_early_stopped = True  # Skip Phase 1
            self.selector.on_phase2_start()     # Initialize warmup counter
        elif self.adv_pretrained_path:
            logger.warning(
                "[FedCQD] adv_pretrained_path='%s' not found, "
                "will train Phase 1 from scratch.",
                self.adv_pretrained_path)

        # ── Load per-factor pretrained heads ────────────────────────
        # When adv_pretrained_heads is provided, load each factor's head
        # from its own Phase 1 checkpoint. This allows combining heads
        # that were independently pretrained on different bias factors,
        # and each head can have its own GRL lambda.
        if self.adv_pretrained_heads and not self._adv_loaded_from_disk:
            loaded_factors = self._load_pretrained_heads(device)
            if loaded_factors:
                self._adv_loaded_from_disk = True
                self._phase1_early_stopped = True
                self.selector.on_phase2_start()
                logger.info(
                    "[FedCQD] Loaded per-factor pretrained heads: %s",
                    loaded_factors)

        # ── LoRA layer selection ──────────────────────────────────────
        self._extra_lr = config.train.optimizer.lr if not only_for_eval else None

        raw_layer_ids = getattr(config.trainer, 'trainable_lora_layer_ids', [])
        self.trainable_lora_layer_ids = (
            list(raw_layer_ids) if raw_layer_ids else [])
        self.top_k_lora_layers = int(
            getattr(config.trainer, 'top_k_lora_layers', 4))

        logger.info(
            "[FedCQD] hidden=%d selector=%s adv_pretrain_rounds=%d "
            "adv_pretrained_path=%s loaded_from_disk=%s "
            "top_k_lora_layers=%d explicit_lora_layers=%s",
            self.hidden_size, self.causal_selector_type,
            self.adv_pretrain_rounds,
            self.adv_pretrained_path or '(none)',
            self._adv_loaded_from_disk,
            self.top_k_lora_layers, self.trainable_lora_layer_ids
        )

        # ── Phase management ──────────────────────────────────────────
        self.current_phase = None
        self.current_round_idx = 0
        self._phase_applied = None
        self.local_train_round = 0
        self._phase1_early_stopped = False

        self.all_lora_layer_ids = self._discover_lora_layer_ids()
        self.selected_lora_layer_ids = self._select_lora_layer_ids()

        logger.info(
            "[FedCQD] discovered LoRA layers=%s | "
            "selected Phase-2 LoRA layers=%s",
            self.all_lora_layer_ids, self.selected_lora_layer_ids
        )

        # ── Adv test split ────────────────────────────────────────────
        self._adv_test_batches = []
        self._adv_test_indices = set()
        self._adv_train_indices = set()
        self._phase1_sample_cursor = 0

        # ── Running metrics accumulation ──────────────────────────────
        self._running_metrics = {}

    # ── Federated aggregation: include score_head + selector ──────────
    # In federated training, GeneralTorchTrainer.get_model_para() only
    # returns ctx.model.state_dict().  score_head and selector are
    # independent nn.Module attributes that do NOT appear in
    # ctx.model.state_dict(), so they are never aggregated across
    # clients.  This causes 53 clients to train divergent score_heads
    # and selectors, breaking federated convergence.
    #
    # Fix: override get_model_para() / update() to encode score_head
    # and selector weights into the transmitted state dict.
    #
    # Each extra parameter is stored as a SEPARATE top-level key with a
    # reserved prefix so that ClientsAvgAggregator._para_weighted_avg()
    # can treat them as ordinary tensors (weighted averaging).  Nested
    # state_dicts would break the aggregator (OrderedDict * float fails).
    #
    # Key format:
    #   __sc__{param_name}  → score_head parameter (tensor)
    #   __sel__{param_name} → selector parameter (tensor)
    #
    # update() reconstructs state_dicts from these flat keys.

    _SC_PREFIX = '__sc__'
    _SEL_PREFIX = '__sel__'

    @staticmethod
    def _flatten_state_dict(state_dict, prefix):
        """Convert {name: tensor} → {prefix+name: tensor}."""
        return {f'{prefix}{k}': v for k, v in state_dict.items()}

    @staticmethod
    def _unflatten_state_dict(flat_dict, prefix):
        """Extract {prefix+name: tensor} → {name: tensor}."""
        n = len(prefix)
        return {
            k[n:]: v for k, v in flat_dict.items() if k.startswith(prefix)
        }

    def get_model_para(self):
        """Override to include score_head and selector in federated comm."""
        state = super().get_model_para()
        # Flatten score_head and selector into individual tensor keys
        state.update(self._flatten_state_dict(
            self.score_head.state_dict(), self._SC_PREFIX))
        state.update(self._flatten_state_dict(
            self.selector.state_dict(), self._SEL_PREFIX))
        return state

    def update(self, model_parameters, strict=False):
        """Override to extract and load score_head and selector."""
        # Extract score_head and selector parameters BEFORE parent processes
        score_head_state = self._unflatten_state_dict(
            model_parameters, self._SC_PREFIX)
        selector_state = self._unflatten_state_dict(
            model_parameters, self._SEL_PREFIX)

        # Remove prefixed keys so parent doesn't see them
        for k in list(model_parameters.keys()):
            if k.startswith(self._SC_PREFIX) or k.startswith(self._SEL_PREFIX):
                del model_parameters[k]

        # Parent handles ctx.model (LLM + LoRA)
        super().update(model_parameters, strict=strict)

        # Load score_head and selector from aggregated (weighted-averaged)
        # parameters.  The aggregator treats each __sc__* and __sel__* key
        # as an independent tensor and performs element-wise weighted avg.
        if score_head_state:
            self.score_head.load_state_dict(score_head_state, strict=True)
        if selector_state:
            self.selector.load_state_dict(selector_state, strict=True)

    # ── LoRA layer discovery ──────────────────────────────────────────

    def _discover_lora_layer_ids(self):
        ids = set()
        patterns = [
            re.compile(r'\.layers\.(\d+)\.'),
            re.compile(r'\.h\.(\d+)\.'),
            re.compile(r'\.block\.(\d+)\.'),
        ]
        for name, _ in self.ctx.model.named_parameters():
            if 'lora_' not in name:
                continue
            for pat in patterns:
                m = pat.search(name)
                if m:
                    ids.add(int(m.group(1)))
                    break
        return sorted(ids)

    def _select_lora_layer_ids(self):
        if not self.all_lora_layer_ids:
            logger.warning('[FedCQD] No LoRA layer ids discovered.')
            return []

        if self.trainable_lora_layer_ids:
            selected = [i for i in self.trainable_lora_layer_ids
                        if i in self.all_lora_layer_ids]
            if not selected:
                raise RuntimeError(
                    '[FedCQD] trainable_lora_layer_ids provided, '
                    'but none exist.')
            return sorted(selected)

        k = max(1, min(self.top_k_lora_layers, len(self.all_lora_layer_ids)))
        return sorted(self.all_lora_layer_ids[-k:])

    # ── Parameter freezing ────────────────────────────────────────────

    def _freeze_all_model_params(self):
        for p in self.ctx.model.parameters():
            p.requires_grad = False

    def _enable_selected_lora_only(self):
        self._freeze_all_model_params()
        selected = set(self.selected_lora_layer_ids)

        for name, p in self.ctx.model.named_parameters():
            if 'lora_' not in name:
                continue
            matched = False
            for lid in selected:
                if (f'.layers.{lid}.' in name or
                    f'.h.{lid}.' in name or
                    f'.block.{lid}.' in name):
                    matched = True
                    break
            if matched:
                p.requires_grad = True

    def _set_module_trainable(self, module, trainable: bool):
        for p in module.parameters():
            p.requires_grad = trainable

    # ── Phase management (enhanced with early stopping) ───────────────

    def _decide_phase(self, round_idx, cur_mode):
        if cur_mode not in ('train', 'finetune'):
            return 2
        # If Phase1 early stopped, go straight to Phase2
        if self._phase1_early_stopped:
            return 2
        return 1 if round_idx < self.adv_pretrain_rounds else 2

    def _activate_phase(self, phase, round_idx=-1):
        if self._phase_applied == phase:
            return

        self._phase_applied = phase

        if phase == 1:
            self._freeze_all_model_params()
            self._set_module_trainable(self.score_head, False)
            # Selector handles its own trainable state
            for p in self.selector.parameters():
                p.requires_grad = True
            logger.info(
                '[FedCQD] >>> Phase 1 activated at round %s: '
                'pretrain adversarial selector only', round_idx)

        elif phase == 2:
            if self.freeze_llm:
                # Ablation: freeze entire LLM, only train score_head
                self._freeze_all_model_params()
                self._set_module_trainable(self.score_head, True)
                for p in self.selector.parameters():
                    p.requires_grad = False
                logger.info(
                    '[FedCQD] >>> Phase 2 activated at round %s: '
                    'FREEZE_LLM ablation — train score_head ONLY (no LoRA)',
                    round_idx)
            else:
                self._enable_selected_lora_only()
                self._set_module_trainable(self.score_head, True)
                if self.minmax_mode:
                    # Min-max mode: selector is also trainable in Phase 2.
                    # All three components trained simultaneously:
                    # LLM+score_head minimize reward_loss + GRL deconfounding,
                    # selector maximizes adversarial prediction (Step 3 enabled).
                    for p in self.selector.parameters():
                        p.requires_grad = True
                    logger.info(
                        '[FedCQD] >>> Phase 2 activated at round %s: '
                        'MIN-MAX mode — train LoRA + score_head + selector (ALL)',
                        round_idx)
                else:
                    # Freeze selector (adversarial head) in Phase 2.
                    # In Stage 2 the adversarial head acts as a fixed "judge":
                    # its GRL gradients flow back to the LLM for deconfounding,
                    # but the selector itself should NOT be updated.
                    for p in self.selector.parameters():
                        p.requires_grad = False
                    logger.info(
                        '[FedCQD] >>> Phase 2 activated at round %s: '
                        'train reward head + partial LoRA layers=%s + GRL selector (FROZEN)',
                        round_idx, self.selected_lora_layer_ids)
        else:
            raise ValueError(f'[FedCQD] Unsupported phase={phase}')

    # ── Adv test split collection ─────────────────────────────────────

    def _collect_adv_test_split(self, ctx):
        """Walk the train dataloader once to assign sample-level indices.

        Global sample index = position in the flattened stream of batches.
        Indices [0, train_end) -> train split.
        Indices [train_end, test_end) -> test split.
        Remaining samples beyond test_end are also used for training.
        Dynamically adjusts when client data < ADV_TRAIN_SIZE.
        """
        if getattr(ctx.cfg.data, 'pool_prepared_train', False):
            from federatedscope.core.auxiliaries.dataloader_builder import (
                get_dataloader,
            )
            from federatedscope.core.data.wrap_dataset import WrapDataset

            loader = getattr(ctx, 'val_loader', None)
            if loader is None:
                dataset = getattr(ctx, 'val_data', None)
                if dataset is None or not len(dataset):
                    raise ValueError(
                        'Pooled head pretraining requires separate validation data')
                if isinstance(dataset, dict):
                    dataset = WrapDataset(dataset)
                loader = get_dataloader(dataset, ctx.cfg, split='val')
            # ReIterator cycles forever; use its finite underlying loader.
            loader = getattr(loader, 'loader', loader)
            self._adv_test_batches = [
                {key: value.detach().cpu().clone()
                 if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
                for batch in loader
            ]
            if not self._adv_test_batches:
                raise ValueError('Pooled head pretraining validation is empty')
            self._adv_train_indices = set()
            self._adv_test_indices = set()
            for factor in self.adv_factors:
                if any(factor not in batch for batch in self._adv_test_batches):
                    raise ValueError(
                        f'Missing {factor} labels in pretraining validation')
                labels = torch.cat([
                    batch[factor].flatten()
                    for batch in self._adv_test_batches])
                if not torch.all((labels == 0) | (labels == 1)):
                    raise ValueError(
                        f'Non-binary {factor} labels in pretraining validation')
                ones = int((labels == 1).sum())
                zeros = labels.numel() - ones
                logger.info(
                    '[FedCQD] pooled validation %s: zero=%d one=%d '
                    'majority_baseline=%.6f',
                    factor, zeros, ones, max(zeros, ones) / labels.numel())
            logger.info(
                '[FedCQD] centralized adv holdout: independent validation, '
                '%d samples; no positional training exclusions.',
                sum(batch['win_input_ids'].size(0)
                    for batch in self._adv_test_batches))
            return

        train_end = ADV_TRAIN_SIZE
        test_end = ADV_TRAIN_SIZE + ADV_TEST_SIZE

        self._adv_train_indices = set()
        self._adv_test_indices = set()
        test_batches = []

        # First pass: count total samples
        total_samples = len(ctx.train_loader.dataset)
        # Adjust splits when client data is insufficient
        if total_samples < ADV_TRAIN_SIZE + ADV_TEST_SIZE:
            actual_test = max(100, int(total_samples * 0.1))
            actual_train = total_samples - actual_test
            train_end = actual_train
            test_end = total_samples
            logger.info(
                "[FedCQD] Adjusted adv split for %d samples: "
                "train_end=%d, test_end=%d",
                total_samples, train_end, test_end,
            )

        global_idx = 0
        for batch in ctx.train_loader:
            bs = batch['win_input_ids'].size(0)
            batch_start = global_idx
            batch_end = global_idx + bs

            # check overlap with test window [train_end, test_end)
            overlap_start = max(batch_start, train_end)
            overlap_end = min(batch_end, test_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - batch_start
                local_end = overlap_end - batch_start
                sliced = {
                    k: v[local_start:local_end]
                    for k, v in batch.items()
                    if hasattr(v, '__getitem__')
                }
                test_batches.append(sliced)
                for gi in range(overlap_start, overlap_end):
                    self._adv_test_indices.add(gi)

            # check overlap with train window [0, train_end)
            tr_overlap_start = max(batch_start, 0)
            tr_overlap_end = min(batch_end, train_end)
            if tr_overlap_start < tr_overlap_end:
                for gi in range(tr_overlap_start, tr_overlap_end):
                    self._adv_train_indices.add(gi)

            global_idx = batch_end
            if global_idx >= test_end:
                break

        self._adv_test_batches = test_batches
        logger.info(
            "[FedCQD] adv split: train=%d samples, test=%d samples "
            "(%d batches)",
            len(self._adv_train_indices),
            len(self._adv_test_indices),
            len(test_batches),
        )

    # ── Adv test evaluation (delegated to selector) ───────────────────

    def _run_adv_test(self, ctx):
        """Evaluate adversarial selector on held-out test split."""
        if not self._adv_test_batches or not self.adv_factors:
            return 0.0

        dev = ctx.device

        def encode_fn(batch):
            win_ids = batch['win_input_ids'].to(dev)
            win_mask = batch['win_attention_mask'].to(dev)
            lose_ids = batch['lose_input_ids'].to(dev)
            lose_mask = batch['lose_attention_mask'].to(dev)
            return self._encode_pair(
                ctx, win_ids, win_mask, lose_ids, lose_mask, require_grad=False,
                win_labels=batch.get('win_labels'), lose_labels=batch.get('lose_labels'))

        result = self.selector.evaluate(
            self._adv_test_batches, encode_fn, dev)
        test_adv_acc = result.get('test_adv_acc', 0.0)
        per_factor = result.get('per_factor', {})

        logger.info(
            "[FedCQD] Phase-1 adv test | test_adv_acc=%.4f | "
            "per_factor=%s",
            test_adv_acc, per_factor
        )

        # Write to eval_results.log
        outdir = getattr(self.cfg, 'outdir', '.')
        log_path = os.path.join(outdir, 'eval_results.log')
        with open(log_path, 'a') as f:
            f.write(
                f"[FedCQD][Phase-1 adv test] "
                f"round={self.current_round_idx} "
                f"test_adv_acc={test_adv_acc:.4f} "
                f"per_factor={per_factor}\n"
            )

        # Check early stopping
        patience = getattr(self.selector, 'adv_early_stop_patience', 0)
        if self.selector.should_early_stop(patience):
            self._phase1_early_stopped = True
            logger.info(
                "[FedCQD] Phase1 early stopping triggered "
                "(no improvement for %d eval rounds).", patience
            )

        return test_adv_acc

    # ── Save pretrained adversarial head ────────────────────────────────

    def _save_best_pooled_head(self, ctx, metrics):
        """Select pooled Phase1 heads using independent validation only."""
        if (not getattr(ctx.cfg.data, 'pool_prepared_train', False)
                or getattr(ctx, 'cur_split', None) != 'val'
                or ctx.cur_mode in ('train', 'finetune')):
            return False
        accuracies = {
            factor: metrics.get(f'adv_{factor}_acc')
            for factor in self.adv_factors
        }
        if (not accuracies or any(
                value is None or not np.isfinite(value)
                or not 0.0 <= value <= 1.0
                for value in accuracies.values())
                or metrics.get('adv_total', 0) <= 0):
            raise ValueError('Missing or invalid Phase1 validation metrics')
        score = sum(accuracies.values()) / len(accuracies)
        previous = getattr(self, '_pooled_best_checkpoint', None)
        if previous is not None and score <= previous['validation_mean_accuracy']:
            return False
        state = {key: value.detach().cpu().clone()
                 for key, value in self.selector.state_dict().items()}
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Non-finite Phase1 head at validation checkpoint')
        self._pooled_best_checkpoint = {
            'selector': state,
            'trained_rounds': self.local_train_round,
            'checkpoint_kind': 'best_validation',
            'selection_rule': 'mean_per_factor_validation_accuracy',
            'best_validation_round': self.current_round_idx,
            'validation_mean_accuracy': score,
            'validation_accuracies': accuracies,
            'validation_samples': int(metrics['adv_total']),
        }
        self._save_adv_pretrained(ctx)
        logger.info(
            '[FedCQD] best pooled Phase1 head: round=%d val=%s mean=%.6f',
            self.current_round_idx, accuracies, score)
        return True

    def _save_adv_pretrained(self, ctx):
        """Save selector to disk after Phase 1 completes.

        The checkpoint is saved to {outdir}/adv_pretrained.pt and can be
        loaded in future runs via trainer.adv_pretrained_path to skip
        Phase 1 entirely.

        Note: score_head is NOT saved because Phase 1 freezes it — its
        weights are just random init.  Phase 2 trains score_head from
        scratch, so loading random weights is meaningless.

        Saved contents:
          - selector: state_dict of the adversarial selector
          - adv_factors: list of bias factor names
          - hidden_size: LLM hidden dimension
          - trained_rounds: number of Phase 1 rounds completed
          - causal_selector_type: selector class identifier
        """
        outdir = getattr(self.cfg, 'outdir', '.')
        save_path = os.path.join(outdir, 'adv_pretrained.pt')

        ckpt = {
            'selector': self.selector.state_dict(),
            'adv_factors': self.adv_factors,
            'hidden_size': self.hidden_size,
            'trained_rounds': self.local_train_round,
            'causal_selector_type': self.causal_selector_type,
            'adv_pooling': getattr(getattr(ctx.cfg, 'trainer', None), 'adv_pooling', 'last'),
        }
        if hasattr(self.selector, 'head_architecture'):
            ckpt['adv_head_architecture'] = dict(self.selector.head_architecture)
        if getattr(ctx.cfg.data, 'require_complete_pair_inputs', False):
            ckpt['pair_input_policy'] = {
                'require_complete': True,
                'context_limit': int(ctx.cfg.llm.tok_len),
            }
        if getattr(ctx.cfg.data, 'pool_prepared_train', False):
            best = getattr(self, '_pooled_best_checkpoint', None)
            if best is None:
                raise ValueError('Pooled Phase1 has no validated head checkpoint')
            ckpt.update(best)

        os.makedirs(outdir, exist_ok=True)
        torch.save(ckpt, save_path)
        logger.info(
            "[FedCQD] Saved pretrained adversarial head to %s "
            "(adv_factors=%s, trained_rounds=%d)",
            save_path, self.adv_factors, self.local_train_round,
        )

    def _load_pretrained_heads(self, device):
        """Load per-factor pretrained adversarial heads from separate
        Phase 1 checkpoints.

        Each checkpoint may contain a selector trained on a single factor
        (e.g., adv_factors=['bias_length']) or multiple factors.  We
        extract only the head(s) matching our target factor and load them
        into the corresponding slot in self.selector.adv_heads[i].

        Args:
            device: torch device to map tensors to.

        Returns:
            List of factor names that were successfully loaded.
        """
        loaded = []
        for factor, path in self.adv_pretrained_heads.items():
            # Find the slot index for this factor in our selector
            if factor not in self.adv_factors:
                logger.warning(
                    "[FedCQD] adv_pretrained_heads has factor '%s' "
                    "not in adv_factors=%s, skipping.",
                    factor, self.adv_factors)
                continue
            slot_idx = self.adv_factors.index(factor)

            if not os.path.isfile(path):
                logger.warning(
                    "[FedCQD] adv_pretrained_heads['%s']='%s' "
                    "not found, this head will train from scratch.",
                    factor, path)
                continue

            ckpt = torch.load(path, map_location=device)
            ckpt_factors = ckpt.get('adv_factors', [])
            selector_state = ckpt.get('selector', None)
            if selector_state is None:
                logger.warning(
                    "[FedCQD] No 'selector' key in %s, skipping.", path)
                continue

            # Find the source head index for this factor in the checkpoint
            if factor in ckpt_factors:
                src_idx = ckpt_factors.index(factor)
            elif len(ckpt_factors) == 1:
                # Single-factor checkpoint: assume it's for this factor
                src_idx = 0
                logger.info(
                    "[FedCQD] Checkpoint %s has single factor %s, "
                    "mapping to target factor %s.",
                    path, ckpt_factors[0], factor)
            else:
                logger.warning(
                    "[FedCQD] Factor '%s' not found in checkpoint %s "
                    "(ckpt_factors=%s), skipping.",
                    factor, path, ckpt_factors)
                continue

            # Extract head_{src_idx} parameters from selector state_dict
            head_prefix = f'adv_heads.{src_idx}.'
            head_state = {}
            for k, v in selector_state.items():
                if k.startswith(head_prefix):
                    new_key = k.replace(
                        head_prefix, f'adv_heads.{slot_idx}.')
                    head_state[new_key] = v

            if not head_state:
                logger.warning(
                    "[FedCQD] No head parameters found for %s in %s, "
                    "skipping.", factor, path)
                continue

            # Load into selector (strict=False to only update this head)
            missing, unexpected = self.selector.load_state_dict(
                head_state, strict=False)
            if missing:
                logger.debug(
                    "[FedCQD] load_state_dict missing keys (expected "
                    "for partial load): %s", missing[:3])
            if unexpected:
                logger.warning(
                    "[FedCQD] load_state_dict unexpected keys: %s",
                    unexpected[:3])

            loaded.append(factor)
            logger.info(
                "[FedCQD] Loaded head[%d] for factor '%s' from %s "
                "(src_idx=%d, ckpt_factors=%s)",
                slot_idx, factor, path, src_idx, ckpt_factors)

        return loaded

    # ── Encoding helper ───────────────────────────────────────────────

    def _encode_pair(self, ctx, win_ids, win_mask, lose_ids, lose_mask,
                     require_grad=True, win_labels=None, lose_labels=None):
        # Bypass lm_head to save ~2.5GB logits tensor per forward pass.
        # Model chain: AdapterModel -> PeftModel -> LoraModel -> CausalLM -> Backbone
        backbone = ctx.model              # AdapterModel
        if hasattr(backbone, 'model'):
            backbone = backbone.model     # PeftModel
        if hasattr(backbone, 'base_model'):
            backbone = backbone.base_model  # LoraModel
        if hasattr(backbone, 'model'):
            backbone = backbone.model     # Qwen2ForCausalLM
        if hasattr(backbone, 'model') and not isinstance(backbone, tuple):
            backbone = backbone.model     # Qwen2Model (skip lm_head)

        if require_grad:
            win_out = backbone(input_ids=win_ids, attention_mask=win_mask)
            lose_out = backbone(input_ids=lose_ids, attention_mask=lose_mask)
        else:
            with torch.no_grad():
                win_out = backbone(input_ids=win_ids, attention_mask=win_mask)
                lose_out = backbone(input_ids=lose_ids, attention_mask=lose_mask)

        cfg = getattr(ctx, 'cfg', None)
        mode = getattr(getattr(cfg, 'trainer', None), 'adv_pooling', 'last')
        if mode != 'last':
            if not getattr(getattr(cfg, 'data', None), 'require_complete_pair_inputs', False):
                raise ValueError('Response pooling requires complete pair inputs')
            batch = getattr(ctx, 'data_batch', {})
            win_labels = batch.get('win_labels') if win_labels is None else win_labels
            lose_labels = batch.get('lose_labels') if lose_labels is None else lose_labels
        win_h = pool_adversarial_hidden(win_out.last_hidden_state, win_mask, win_labels, mode)
        lose_h = pool_adversarial_hidden(lose_out.last_hidden_state, lose_mask, lose_labels, mode)
        return win_h.float(), lose_h.float()

    def _batch_has_all_adv_factors(self, ctx):
        if len(self.adv_factors) == 0:
            return False
        return all(factor in ctx.data_batch for factor in self.adv_factors)

    def _compute_selector_direct_loss(self, z_detached, data_batch, device):
        """Compute selector direct loss with detached input.

        Uses detached z so gradients only flow to selector parameters,
        not back to the LLM. This is used in Phase 2's Step 3 backward
        to keep the selector's discriminator ability alive without
        interfering with the GRL reversal signal.

        Args:
            z_detached: Detached input tensor [B, input_size] (no grad
                        connection to LLM)
            data_batch: Dict containing bias factor labels
            device: torch device

        Returns:
            loss tensor (gradients flow to selector params only)
        """
        adv_loss = torch.tensor(0.0, device=device)

        for i, factor in enumerate(self.adv_factors):
            if factor not in data_batch:
                continue
            lbl = data_batch[factor].float().to(device)
            if lbl.dim() == 1:
                lbl = lbl.unsqueeze(1)

            smooth_lbl = self.selector._smooth_labels(
                lbl, self.selector.adv_label_smoothing)

            # Forward through adv_head WITHOUT GRL (use_grl=False)
            pred = self.selector.adv_heads[i](
                z_detached, lambda_=1.0, use_grl=False)
            this_loss = F.binary_cross_entropy(pred, smooth_lbl)
            adv_loss = adv_loss + this_loss

        return adv_loss

    # ── Hook: fit start (enhanced optimizer setup) ────────────────────

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

        # Reset running metrics for this round
        self._running_metrics = {}

        # ── Set up optimizer param groups ─────────────────────────────
        if self._extra_lr is not None and ctx.cur_mode == 'train':
            # Score head parameters (main lr) -> goes to ctx.optimizer
            score_params = list(self.score_head.parameters())

            # Find existing param ids to avoid duplicates
            existing = {
                id(p)
                for g in ctx.optimizer.param_groups
                for p in g['params']
            }

            # Add score_head with main lr
            new_score = [p for p in score_params if id(p) not in existing]
            if new_score:
                ctx.optimizer.add_param_group({
                    'params': new_score,
                    'lr': self._extra_lr
                })

            # ── Create INDEPENDENT optimizer for selector ───────────────
            # This is CRITICAL to prevent AdamW momentum contamination.
            # Previously, a single optimizer managed both reward model and
            # selector, causing momentum pollution across the two components.
            # Now we use two separate optimizers.
            selector_params = list(self.selector.parameters())
            adv_lr = getattr(self.selector, 'adv_lr', self._extra_lr)
            adv_wd = getattr(self.selector, 'adv_weight_decay',
                             ctx.cfg.train.optimizer.weight_decay
                             if hasattr(ctx.cfg.train, 'optimizer') else 0.0)

            if selector_params and not hasattr(self, '_selector_optimizer'):
                self._selector_optimizer = build_independent_optimizer(
                    ctx.optimizer,
                    selector_params,
                    lr=adv_lr,
                    weight_decay=adv_wd)
                logger.info(
                    '[FedCQD] Created independent selector_optimizer '
                    '(lr=%.2e, wd=%.2e) to prevent momentum contamination.',
                    adv_lr, adv_wd)

            # Store reference for easy access
            ctx._selector_optimizer = getattr(
                self, '_selector_optimizer', None)

        # ── Phase decision ────────────────────────────────────────────
        if ctx.cur_mode in ('train', 'finetune'):
            round_idx = self.local_train_round
            self.current_round_idx = round_idx
            self.current_phase = self._decide_phase(
                round_idx, ctx.cur_mode)
            self._activate_phase(self.current_phase, round_idx)

            # Collect held-out adv test batches on first phase-1 round
            if (self.current_phase == 1
                    and not self._adv_test_batches):
                self._collect_adv_test_split(ctx)

            # Reset per-epoch sample cursor for phase-1 training cap
            self._phase1_sample_cursor = 0
        else:
            if self.current_phase is None:
                self.current_phase = 2

    # ── Hook: batch forward (enhanced with selector delegation) ───────

    def _hook_on_batch_forward(self, ctx):
        if getattr(ctx.cfg.data, 'pool_prepared_train', False):
            # The frozen encoder stays deterministic; heads train only on train.
            ctx.model.eval()
            self.selector.train(ctx.cur_mode in ('train', 'finetune'))
        dev = ctx.device
        win_ids = ctx.data_batch['win_input_ids'].to(dev)
        win_mask = ctx.data_batch['win_attention_mask'].to(dev)
        lose_ids = ctx.data_batch['lose_input_ids'].to(dev)
        lose_mask = ctx.data_batch['lose_attention_mask'].to(dev)
        B = win_ids.size(0)

        phase = self.current_phase if self.current_phase is not None else 2
        if ctx.cur_mode not in ('train', 'finetune'):
            phase = 2

        # ── Phase-1: filter to train split only ───────────────────────
        if phase == 1 and ctx.cur_mode in ('train', 'finetune'):
            cursor = self._phase1_sample_cursor
            batch_global_indices = list(range(cursor, cursor + B))
            self._phase1_sample_cursor += B

            if self._adv_train_indices:
                # Keep samples in train window OR beyond test window
                keep_local = [
                    i for i, gi in enumerate(batch_global_indices)
                    if gi in self._adv_train_indices
                    or gi >= ADV_TRAIN_SIZE + ADV_TEST_SIZE
                ]
                if not keep_local:
                    ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                    ctx.y_true = CtxVar(
                        torch.ones(B, dtype=torch.long, device=dev),
                        LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(
                        torch.zeros(B, dtype=torch.long, device=dev),
                        LIFECYCLE.BATCH)
                    ctx.loss_batch = CtxVar(
                        torch.tensor(0.0, device=dev, requires_grad=True),
                        LIFECYCLE.BATCH)
                    ctx.batch_size = CtxVar(B, LIFECYCLE.BATCH)
                    return
                if len(keep_local) < B:
                    idx_t = torch.tensor(
                        keep_local, dtype=torch.long, device=dev)
                    win_ids = win_ids[idx_t]
                    win_mask = win_mask[idx_t]
                    lose_ids = lose_ids[idx_t]
                    lose_mask = lose_mask[idx_t]
                    for key in list(ctx.data_batch.keys()):
                        val = ctx.data_batch[key]
                        if (hasattr(val, '__getitem__')
                                and hasattr(val, 'size')):
                            try:
                                ctx.data_batch[key] = val[idx_t.cpu()]
                            except Exception:
                                pass
                    B = win_ids.size(0)

        has_adv_labels = self._batch_has_all_adv_factors(ctx)

        # Pre-forward NaN check: if score_head or LoRA parameters are already
        # NaN (from a previous catastrophic update), reset them and skip.
        if phase == 2 and ctx.cur_mode in ('train', 'finetune'):
            param_nan = False
            for p in self.score_head.parameters():
                if torch.isnan(p).any():
                    param_nan = True
                    break
            if not param_nan:
                for name, p in ctx.model.named_parameters():
                    if p.requires_grad and torch.isnan(p).any():
                        param_nan = True
                        logger.error(
                            '[FedCQD] NaN in LoRA param before forward: %s',
                            name)
                        break
            if param_nan:
                # Reset score_head weights
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.score_head.weight)
                    nn.init.zeros_(self.score_head.bias)
                # Zero out NaN LoRA params
                for name, p in ctx.model.named_parameters():
                    if p.requires_grad and torch.isnan(p).any():
                        with torch.no_grad():
                            p.zero_()
                        logger.warning(
                            '[FedCQD] Reset NaN params, skipping batch')
                ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                ctx.y_true = CtxVar(
                    torch.ones(B, dtype=torch.long, device=dev),
                    LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(
                    torch.zeros(B, dtype=torch.long, device=dev),
                    LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(
                    torch.tensor(0.0, device=dev, requires_grad=True),
                    LIFECYCLE.BATCH)
                ctx.batch_size = CtxVar(B, LIFECYCLE.BATCH)
                return

        # ── Phase 1: pretrain adversarial selector ────────────────────
        if phase == 1:
            win_h, lose_h = self._encode_pair(
                ctx, win_ids, win_mask, lose_ids, lose_mask,
                require_grad=False
            )

            if has_adv_labels:
                adv_loss, metrics = self.selector.compute_loss(
                    win_h, lose_h, ctx.data_batch,
                    use_grl=False, device=dev
                )
                total_loss = adv_loss
                self._accumulate_metrics(metrics, B)
            else:
                # Extreme case: no bias labels in this batch
                total_loss = torch.tensor(
                    0.0, device=dev, requires_grad=True)

            with torch.no_grad():
                win_score = self.score_head(win_h)
                lose_score = self.score_head(lose_h)
                preds = (win_score > lose_score).squeeze(1).long()

        # ── Phase 2: joint training (reward + GRL selector) ──────────
        else:
            require_grad = ctx.cur_mode in ('train', 'finetune')

            if require_grad and has_adv_labels and any(
                    v > 0 for v in self.selector.grl_lambdas.values()):
                # ── SEPARATE FORWARD PASSES ──────────────────────────
                # Key insight: GRL NaN was caused by shared computation
                # graph between reward_loss and adv_loss_grl.  When
                # reward_loss.backward(retain_graph=True) is called first,
                # the GRL autograd node's intermediate state becomes
                # inconsistent, producing NaN gradients on the second
                # backward pass.
                #
                # Fix: Use TWO independent forward passes so that
                # reward_loss and adv_loss_grl have completely separate
                # computation graphs.  This eliminates the need for
                # retain_graph=True and makes GRL work correctly.
                #
                # Overhead: one extra LLM forward pass (~50-100ms on 3090
                # for Qwen1.5-0.5B), which is negligible compared to the
                # training stability gains.

                # Forward Pass 1: Reward path (independent graph)
                win_h_1, lose_h_1 = self._encode_pair(
                    ctx, win_ids, win_mask, lose_ids, lose_mask,
                    require_grad=True
                )
                win_score = self.score_head(win_h_1)
                lose_score = self.score_head(lose_h_1)
                score_diff = (win_score - lose_score).clamp(
                    min=-10.0, max=10.0)
                reward_loss = -F.logsigmoid(score_diff).mean()

                # Progressive lambda warmup — compute per-factor effective
                # lambdas so each adversarial head has its own warmup-scaled
                # GRL lambda.  This unifies warmup control and gradient
                # direction into the GRL layer itself (Plan B fix).
                phase2_round = self.selector._phase2_round_counter
                factor_effective_lambdas = {}
                for factor in self.selector.adv_factors:
                    base_lambda = self.selector.grl_lambdas.get(factor, 0.0)
                    if base_lambda <= 0:
                        factor_effective_lambdas[factor] = 0.0
                    else:
                        warmup_factor = min(
                            1.0,
                            (phase2_round + 1) / self.selector.adv_warmup_rounds
                        ) if self.selector.adv_warmup_rounds > 0 else 1.0
                        factor_effective_lambdas[factor] = (
                            base_lambda * warmup_factor)

                # Legacy effective_lambda: max across heads (for total_loss
                # and backward compatibility)
                effective_lambda = (
                    max(factor_effective_lambdas.values())
                    if factor_effective_lambdas else 0.0)

                # Forward Pass 2: GRL + Selector path (independent graph)
                win_h_2, lose_h_2 = self._encode_pair(
                    ctx, win_ids, win_mask, lose_ids, lose_mask,
                    require_grad=True
                )
                # Pass per-factor lambdas to compute_loss so each head's
                # GRL uses its own λ.  The outer effective_lambda is NOT
                # used inside compute_loss — GRL handles both direction
                # (reversal) and strength (λ scaling) internally.
                adv_loss_grl, metrics = self.selector.compute_loss(
                    win_h_2, lose_h_2, ctx.data_batch,
                    use_grl=True, device=dev,
                    factor_lambdas=factor_effective_lambdas,
                )

                # Direct path: use DETACHED hidden states so gradients
                # from this path only update selector parameters, NOT the
                # reward model.  This prevents the selector's normal
                # gradients from interfering with the GRL reversal signal.
                z_detached = self.selector._prepare_input(
                    win_h_2, lose_h_2
                ).detach()
                adv_loss_direct = self._compute_selector_direct_loss(
                    z_detached, ctx.data_batch, dev
                )
                self._accumulate_metrics(metrics, B)

                # Store for split-backward (no retain_graph needed!)
                ctx._reward_loss = reward_loss
                ctx._adv_loss_grl = adv_loss_grl
                ctx._adv_loss_direct = adv_loss_direct
                ctx._effective_lambda = effective_lambda
                ctx._has_adv_labels = True

                total_loss = (
                    reward_loss - effective_lambda * adv_loss_grl)
            else:
                # Eval/test or no adv labels: single forward pass
                win_h, lose_h = self._encode_pair(
                    ctx, win_ids, win_mask, lose_ids, lose_mask,
                    require_grad=require_grad
                )

                win_score = self.score_head(win_h)
                lose_score = self.score_head(lose_h)
                score_diff = (win_score - lose_score).clamp(
                    min=-10.0, max=10.0)
                reward_loss = -F.logsigmoid(score_diff).mean()

                if require_grad:
                    ctx._reward_loss = reward_loss
                    ctx._has_adv_labels = False
                    total_loss = reward_loss
                else:
                    # eval/test: compute adv metrics without affecting loss
                    if has_adv_labels:
                        self.selector.eval()  # disable dropout for eval
                        try:
                            with torch.no_grad():
                                _, metrics = self.selector.compute_loss(
                                    win_h, lose_h, ctx.data_batch,
                                    use_grl=False, device=dev
                                )
                                self._accumulate_metrics(metrics, B)
                        finally:
                            if getattr(ctx.cfg.data, 'pool_prepared_train', False):
                                self.selector.eval()
                            else:
                                self.selector.train()  # legacy mode restoration
                    total_loss = reward_loss

            # Diagnostic: log score statistics periodically
            if require_grad and has_adv_labels:
                with torch.no_grad():
                    # Use win_h_1/lose_h_1 for diagnostics (reward path)
                    score_max = win_score.abs().max().item()
                    score_min = win_score.abs().min().item()
                    diff_mean = score_diff.mean().item()
                    diff_std = score_diff.std().item()
                    if not hasattr(self, '_diag_counter'):
                        self._diag_counter = 0
                    self._diag_counter += 1
                    log_interval = (
                        10 if self._diag_counter <= 200 else 50)
                    if (self._diag_counter <= 5
                            or self._diag_counter % log_interval == 0):
                        logger.info(
                            '[FedCQD][Phase2][diag] batch#%d '
                            'win_score abs_max=%.4f abs_min=%.4f '
                            'diff_mean=%.4f diff_std=%.4f '
                            'reward_loss=%.4f',
                            self._diag_counter, score_max, score_min,
                            diff_mean, diff_std, reward_loss.item())

            # NaN check for reward_loss
            rl_check = reward_loss
            if torch.isnan(rl_check) or torch.isinf(rl_check):
                ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                logger.warning(
                    '[FedCQD] NaN/Inf in reward_loss, skipping batch.')
                ctx.y_true = CtxVar(
                    torch.ones(B, dtype=torch.long, device=dev),
                    LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(
                    torch.zeros(B, dtype=torch.long, device=dev),
                    LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(
                    rl_check.detach().requires_grad_(True),
                    LIFECYCLE.BATCH)
                ctx.batch_size = CtxVar(B, LIFECYCLE.BATCH)
                return

            with torch.no_grad():
                preds = (win_score > lose_score).squeeze(1).long()

        ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(
            torch.ones(B, dtype=torch.long, device=dev), LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(total_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(B, LIFECYCLE.BATCH)

    # ── Metrics accumulation ──────────────────────────────────────────

    def _accumulate_metrics(self, metrics, batch_size=1):
        """Accumulate selector metrics across batches for round-level reporting."""
        for key, value in metrics.items():
            if key not in self._running_metrics:
                self._running_metrics[key] = []
            if getattr(self.cfg.data, 'pool_prepared_train', False):
                value *= batch_size
            self._running_metrics[key].append(value)
        # Track sample count so _collect_selector_metrics can emit adv_total
        if '_total_samples' not in self._running_metrics:
            self._running_metrics['_total_samples'] = 0
        self._running_metrics['_total_samples'] += batch_size

    def _collect_selector_metrics(self):
        """Aggregate accumulated metrics for round-level reporting.

        Also adds ``adv_total`` (total sample count) so that the
        monitor's ``format_eval_res`` does not raise ValueError when
        it encounters ``adv_*`` keys.  The monitor expects every
        ``{prefix}_*`` metric to have a matching ``{prefix}_total``.
        """
        result = {}
        total_samples = self._running_metrics.pop('_total_samples', 0)
        for key, values in self._running_metrics.items():
            if key == '_total_samples':
                continue
            if values:
                denominator = (total_samples if getattr(
                    self.cfg.data, 'pool_prepared_train', False)
                    else len(values))
                result[key] = sum(values) / denominator
        # Required by monitor.format_eval_res: any key starting with "adv_"
        # must have a corresponding "adv_total" entry.
        if any(k.startswith('adv_') for k in result):
            result['adv_total'] = total_samples
        return result

    # ── Hook: batch backward ──────────────────────────────────────────

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cur_mode in ('train', 'finetune'):
            phase = self.current_phase if self.current_phase is not None else 2

            # Phase 2 uses separate backward passes for reward and adversarial
            # components to prevent GRL gradient conflicts from destabilizing
            # AdamW's momentum estimates.
            if (phase == 2
                    and hasattr(ctx, '_reward_loss')
                    and (hasattr(ctx, '_adv_loss_grl')
                         or getattr(ctx, '_has_adv_labels', False))):
                self._phase2_separate_backward(ctx)
                return

            # Default: single backward (Phase 1 or eval)
            ctx.optimizer.zero_grad()
            # Also zero selector optimizer's gradients (selector has its
            # own independent optimizer to avoid momentum contamination
            # with the reward model, but in Phase 1 only selector is
            # trainable, so we must step its optimizer too).
            selector_optim = getattr(self, '_selector_optimizer', None)
            if selector_optim is not None:
                selector_optim.zero_grad()

            ctx.loss_batch.backward()

            # Gradient clipping: always enforce a default max_norm to prevent
            # gradient explosion in Phase 2 (especially with GRL).
            # User can override via cfg.grad.grad_clip in YAML.
            # Default: 1.0 (safe for reward + GRL training)
            grad_clip_val = getattr(ctx, 'grad_clip', -1.0)
            if grad_clip_val is None or grad_clip_val <= 0:
                grad_clip_val = 1.0  # safe default

            trainable_params = []
            for p in ctx.model.parameters():
                if p.requires_grad and p.grad is not None:
                    trainable_params.append(p)
            for p in self.score_head.parameters():
                if p.requires_grad and p.grad is not None:
                    trainable_params.append(p)
            for p in self.selector.parameters():
                if p.requires_grad and p.grad is not None:
                    trainable_params.append(p)

            if trainable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, grad_clip_val)
                # Diagnostic: log gradient norm for first few batches
                if not hasattr(self, '_grad_diag_counter'):
                    self._grad_diag_counter = 0
                self._grad_diag_counter += 1
                if self._grad_diag_counter <= 5 or self._grad_diag_counter % 50 == 0:
                    logger.info(
                        '[FedCQD][backward][diag] batch#%d '
                        'grad_norm=%.4f (clip_to=%.1f)',
                        self._grad_diag_counter,
                        grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        grad_clip_val)

            # Step main optimizer (updates LLM + score_head params;
            # in Phase 1 these are frozen so this is a no-op)
            ctx.optimizer.step()
            # Step selector optimizer (CRITICAL for Phase 1: this is the
            # only optimizer that actually updates trainable parameters
            # during adversarial head pretraining)
            if selector_optim is not None:
                selector_optim.step()

            if ctx.scheduler is not None:
                ctx.scheduler.step()

            # Post-step NaN detection: check if any parameter became NaN
            # after optimizer.step(). If so, log and reset the parameter.
            nan_params = []
            for name, p in ctx.model.named_parameters():
                if p.requires_grad and torch.isnan(p).any():
                    nan_params.append(name)
            for name, p in self.score_head.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(f'score_head.{name}')
            if nan_params:
                logger.error(
                    '[FedCQD] NaN detected in parameters after optimizer.step(): %s. '
                    'This indicates training has diverged.',
                    nan_params[:5]  # log first 5 only
                )

        for key in list(ctx.data_batch.keys()):
            if hasattr(ctx.data_batch[key], 'cpu'):
                ctx.data_batch[key] = ctx.data_batch[key].cpu()

    # ── Phase 2 separate backward (no retain_graph, GRL-safe) ────────

    def _phase2_separate_backward(self, ctx):
        """Three-step backward for Phase 2 with separate forward passes.

        Uses INDEPENDENT computation graphs for reward_loss and
        adv_loss_grl (via separate forward passes), so NO retain_graph
        is needed.  This fixes the GRL NaN problem caused by shared
        autograd nodes under retain_graph=True.

        Step 1: reward_loss → update reward model (score_head + LoRA)
                Computation graph: Forward Pass 1 (independent)
        Step 2: -λ * adv_loss_grl → update reward model (GRL reversal)
                Computation graph: Forward Pass 2 (independent)
                The GRL reverses gradients so the reward model learns to
                produce confounder-free representations.
        Step 3: adv_loss_direct → update selector ONLY (no GRL)
                Computation graph: detached z, gradients only flow to
                selector parameters. Keeps the selector's discriminator
                ability alive, preventing sigmoid saturation.

        Uses independent optimizers for reward model and selector to
        prevent AdamW momentum contamination.
        """
        grad_clip_val = getattr(ctx, 'grad_clip', -1.0)
        if grad_clip_val is None or grad_clip_val <= 0:
            grad_clip_val = 1.0

        reward_loss = ctx._reward_loss
        adv_loss_grl = getattr(ctx, '_adv_loss_grl', None)
        adv_loss_direct = getattr(ctx, '_adv_loss_direct', None)
        effective_lambda = getattr(ctx, '_effective_lambda', 0.0)
        has_adv_labels = getattr(ctx, '_has_adv_labels', False)
        selector_optim = getattr(ctx, '_selector_optimizer', None)

        grl_grad_norm = 0.0  # track for diagnostics

        # ── Step 1: Update reward model with reward_loss ────────────
        # Forward Pass 1's computation graph is independent, so no
        # retain_graph is needed.
        ctx.optimizer.zero_grad()
        reward_loss.backward()

        # NaN diagnostic: detailed check on layer16 lora_A before and after step
        if not hasattr(self, '_step1_diag_counter'):
            self._step1_diag_counter = 0
        self._step1_diag_counter += 1
        if self._step1_diag_counter <= 2:
            for name, p in ctx.model.named_parameters():
                if 'lora_' in name and p.requires_grad:
                    if p.grad is not None:
                        logger.info(
                            '[FedCQD][Step1-grad-all] #%d %s '
                            'param_max=%.6f grad_max=%.6f',
                            self._step1_diag_counter, name,
                            p.data.abs().max().item(),
                            p.grad.abs().max().item())
                    else:
                        logger.info(
                            '[FedCQD][Step1-grad-all] #%d %s grad=None',
                            self._step1_diag_counter, name)

        # DETAILED NaN trace: print lora_A layer16 param values before step
        _diag_lora16a = None
        _diag_lora16a_name = None
        if self._step1_diag_counter <= 2:
            for name, p in ctx.model.named_parameters():
                if name == 'model.base_model.model.model.layers.16.self_attn.q_proj.lora_A.default.weight':
                    _diag_lora16a = p.data.clone()
                    _diag_lora16a_name = name
                    # Print first 5 values
                    flat = p.data.flatten()
                    logger.info(
                        '[FedCQD][Step1-PRE-STEP] #%d %s first5=[%s] '
                        'max=%.8f min=%.8f has_nan=%s',
                        self._step1_diag_counter, name,
                        ', '.join(f'%.8f' % v for v in flat[:5].tolist()),
                        p.data.max().item(), p.data.min().item(),
                        torch.isnan(p.data).any().item())
                    break

        # Clip gradients for reward model params
        reward_params = []
        for p in ctx.model.parameters():
            if p.requires_grad and p.grad is not None:
                reward_params.append(p)
        for p in self.score_head.parameters():
            if p.requires_grad and p.grad is not None:
                reward_params.append(p)

        reward_grad_norm = 0.0
        if reward_params:
            reward_grad_norm = torch.nn.utils.clip_grad_norm_(
                reward_params, grad_clip_val).item()

        # Zero selector gradients before reward step
        for p in self.selector.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # NaN diagnostic: check if gradients are NaN BEFORE optimizer.step()
        pre_step_nan = False
        for name, p in ctx.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    pre_step_nan = True
                    logger.error(
                        '[FedCQD] NaN/Inf in grad BEFORE Step1 step: %s '
                        'grad_nan=%d grad_max=%.4f',
                        name, torch.isnan(p.grad).sum().item(),
                        p.grad.abs().max().item())
                    break
        if not pre_step_nan:
            for p in self.score_head.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        pre_step_nan = True
                        logger.error(
                            '[FedCQD] NaN/Inf in score_head grad BEFORE Step1 step')
                        break

        # NaN diagnostic: check LoRA param values and AdamW state BEFORE step
        if not hasattr(self, '_step1_diag_counter'):
            self._step1_diag_counter = 0
        self._step1_diag_counter += 1
        if self._step1_diag_counter <= 3:
            for name, p in ctx.model.named_parameters():
                if 'lora_A' in name and p.requires_grad:
                    state = ctx.optimizer.state.get(p, {})
                    step_count = state.get('step', 0)
                    exp_avg = state.get('exp_avg', None)
                    exp_avg_sq = state.get('exp_avg_sq', None)
                    logger.info(
                        '[FedCQD][Step1-pre-step][diag] #%d %s '
                        'param_max=%.6f grad_max=%.6f '
                        'exp_avg_max=%.6f exp_avg_sq_max=%.6f step=%s',
                        self._step1_diag_counter, name,
                        p.data.abs().max().item(),
                        p.grad.abs().max().item() if p.grad is not None else -1,
                        exp_avg.abs().max().item() if exp_avg is not None else -1,
                        exp_avg_sq.abs().max().item() if exp_avg_sq is not None else -1,
                        step_count)
                    break  # only log first lora_A

        ctx.optimizer.step()

        # DETAILED NaN trace: print lora_A layer16 param values AFTER step
        if self._step1_diag_counter <= 2 and _diag_lora16a is not None:
            for name, p in ctx.model.named_parameters():
                if name == _diag_lora16a_name:
                    flat = p.data.flatten()
                    logger.info(
                        '[FedCQD][Step1-POST-STEP] #%d %s first5=[%s] '
                        'max=%.8f min=%.8f has_nan=%s',
                        self._step1_diag_counter, name,
                        ', '.join(f'%.8f' % v for v in flat[:5].tolist()),
                        p.data.max().item() if not torch.isnan(p.data).any() else float('nan'),
                        p.data.min().item() if not torch.isnan(p.data).any() else float('nan'),
                        torch.isnan(p.data).any().item())
                    break

        # NaN diagnostic: check if params became NaN AFTER optimizer.step()
        post_step_nan = False
        for name, p in ctx.model.named_parameters():
            if p.requires_grad and torch.isnan(p).any():
                post_step_nan = True
                # Check AdamW state after step
                state = ctx.optimizer.state.get(p, {})
                exp_avg = state.get('exp_avg', None)
                exp_avg_sq = state.get('exp_avg_sq', None)
                step_count = state.get('step', 0)
                logger.error(
                    '[FedCQD] NaN in param AFTER Step1 step #%d: %s '
                    'pre_step_nan=%s adam_step=%s '
                    'exp_avg_nan=%d exp_avg_sq_nan=%d',
                    self._step1_diag_counter, name, pre_step_nan,
                    step_count,
                    torch.isnan(exp_avg).sum().item() if exp_avg is not None else -1,
                    torch.isnan(exp_avg_sq).sum().item() if exp_avg_sq is not None else -1)
                break

        # ── Step 2: GRL reversal path → update reward model ─────────
        # Forward Pass 2's computation graph is independent from
        # Forward Pass 1, so NO retain_graph is needed.
        # The GRL reverses gradients: reward model learns to produce
        # representations that confuse the selector (deconfounding).
        # NOTE: Each adversarial head's GRL uses its own per-factor lambda
        # (set via factor_lambdas in compute_loss), which handles BOTH
        # direction (reversal) and strength (warmup scaling).
        # Do NOT add an outer negative sign — that caused a double-negative
        # bug where GRL reversal and the outer negation cancelled out,
        # making the LLM reinforce bias instead of removing it.
        if has_adv_labels and adv_loss_grl is not None and effective_lambda > 0:
            ctx.optimizer.zero_grad()
            adv_loss_grl.backward()

            # Clip reward model gradients from GRL path
            grl_params = []
            for p in ctx.model.parameters():
                if p.requires_grad and p.grad is not None:
                    grl_params.append(p)
            for p in self.score_head.parameters():
                if p.requires_grad and p.grad is not None:
                    grl_params.append(p)

            if grl_params:
                grl_grad_norm = torch.nn.utils.clip_grad_norm_(
                    grl_params, grad_clip_val).item()

            # NaN detection on GRL gradients (safety check)
            has_nan_grad = False
            for p in ctx.model.parameters():
                if p.requires_grad and p.grad is not None:
                    if torch.isnan(p.grad).any():
                        has_nan_grad = True
                        break
            if not has_nan_grad:
                for p in self.score_head.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan_grad = True
                        break

            if has_nan_grad:
                logger.warning(
                    '[FedCQD] NaN in GRL gradients (Step 2), '
                    'skipping GRL update. This should not happen '
                    'with separate forward passes.')
                # Zero all gradients to prevent NaN from entering optimizer
                ctx.optimizer.zero_grad()
            else:
                # Zero selector gradients before reward model step
                # (selector should NOT be updated by GRL path)
                for p in self.selector.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

                ctx.optimizer.step()

        # ── Step 3: Direct selector update (SKIPPED if selector is frozen) ──
        # In Phase 2 the selector (adversarial head) is frozen: it serves
        # as a fixed judge whose GRL gradients guide the reward model to
        # produce confounder-free representations.  We skip the selector
        # direct-update step because its parameters are not trainable.
        sel_grad_norm = 0.0
        selector_frozen = not any(
            p.requires_grad for p in self.selector.parameters())
        if selector_frozen:
            # Selector is frozen in Phase 2 — skip Step 3 entirely.
            # Just log diagnostics for Step 1 + Step 2.
            if not hasattr(self, '_grad_diag_counter'):
                self._grad_diag_counter = 0
            self._grad_diag_counter += 1
            if self._grad_diag_counter <= 10 or self._grad_diag_counter % 100 == 0:
                logger.info(
                    '[FedCQD][3step-backward][diag] batch#%d '
                    'reward_grad=%.4f grl_grad=%.4f selector=FROZEN '
                    '(clip=%.1f lambda=%.4f)',
                    self._grad_diag_counter,
                    reward_grad_norm,
                    grl_grad_norm,
                    grad_clip_val,
                    effective_lambda)
        elif has_adv_labels and adv_loss_direct is not None and selector_optim is not None:
            selector_optim.zero_grad()
            adv_loss_direct.backward()

            # Clip selector gradients
            selector_params = []
            for p in self.selector.parameters():
                if p.requires_grad and p.grad is not None:
                    selector_params.append(p)

            if selector_params:
                sel_grad_norm = torch.nn.utils.clip_grad_norm_(
                    selector_params, grad_clip_val).item()

            # NaN diagnostic: check selector grads before step
            sel_pre_nan = False
            for p in self.selector.parameters():
                if p.requires_grad and p.grad is not None:
                    if torch.isnan(p.grad).any():
                        sel_pre_nan = True
                        logger.error(
                            '[FedCQD] NaN in selector grad BEFORE Step3 step '
                            'sel_grad_norm=%.4f', sel_grad_norm)
                        break

            # Zero reward model gradients before selector step
            for p in ctx.model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.zero_()
            for p in self.score_head.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            selector_optim.step()

            # Diagnostic
            if not hasattr(self, '_grad_diag_counter'):
                self._grad_diag_counter = 0
            self._grad_diag_counter += 1
            if self._grad_diag_counter <= 10 or self._grad_diag_counter % 100 == 0:
                logger.info(
                    '[FedCQD][3step-backward][diag] batch#%d '
                    'reward_grad=%.4f grl_grad=%.4f selector_grad=%.4f '
                    '(clip=%.1f lambda=%.4f)',
                    self._grad_diag_counter,
                    reward_grad_norm,
                    grl_grad_norm,
                    sel_grad_norm,
                    grad_clip_val,
                    effective_lambda)
        else:
            # No adv data, just reward step
            if not hasattr(self, '_grad_diag_counter'):
                self._grad_diag_counter = 0
            self._grad_diag_counter += 1
            if self._grad_diag_counter <= 5 or self._grad_diag_counter % 50 == 0:
                logger.info(
                    '[FedCQD][reward-only-backward][diag] batch#%d '
                    'reward_grad=%.4f (clip=%.1f)',
                    self._grad_diag_counter,
                    reward_grad_norm,
                    grad_clip_val)

        # ── Scheduler step (only for reward optimizer) ─────────────
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get('loss_regular', 0.0))
        ctx.ys_true.append(ctx.y_true)
        ctx.ys_pred.append(ctx.y_pred)

    # ── Hook: fit end (enhanced with phase transitions) ───────────────

    def _hook_on_fit_end(self, ctx):
        # Safely concatenate ys_true / ys_pred; may be empty when all
        # batches were skipped due to NaN/Inf protection in Phase 2.
        try:
            if ctx.ys_true:
                ctx.ys_true = CtxVar(
                    torch.cat(ctx.ys_true), LIFECYCLE.ROUTINE)
                ctx.ys_pred = CtxVar(
                    torch.cat(ctx.ys_pred), LIFECYCLE.ROUTINE)
        except (RuntimeError, TypeError):
            pass

        # If all batches were skipped (NaN/Inf protection), there are
        # no valid samples.  Set dummy tensors AND a num_samples guard
        # so that monitor.eval() does not crash on empty metrics.
        num_samples = getattr(ctx, 'num_samples', 0)
        if isinstance(ctx.ys_true, list) or num_samples == 0:
            ctx.ys_true = CtxVar(
                torch.tensor([0, 1], dtype=torch.long), LIFECYCLE.ROUTINE)
            ctx.ys_pred = CtxVar(
                torch.tensor([0, 1], dtype=torch.long), LIFECYCLE.ROUTINE)
            ctx.num_samples = CtxVar(1, LIFECYCLE.ROUTINE)
            ctx.loss_batch_total = CtxVar(0.0, LIFECYCLE.ROUTINE)

        results = ctx.monitor.eval(ctx)

        # Add selector metrics to results
        selector_metrics = self._collect_selector_metrics()
        if isinstance(results, dict) and selector_metrics:
            results.update(selector_metrics)

        if selector_metrics:
            phase_name = (
                "Phase-1(adv-pretrain)"
                if self.current_phase == 1 else
                "Phase-2(reward+LoRA/GRL)"
            )
            logger.info(
                "[FedCQD][%s][mode=%s][round=%s] adv_metrics=%s",
                phase_name,
                getattr(ctx, 'cur_mode', 'unknown'),
                self.current_round_idx,
                selector_metrics
            )

        setattr(ctx, 'eval_metrics', results)

        if (getattr(ctx.cfg.data, 'pool_prepared_train', False)
                and self.current_phase == 1 and ctx.cur_split == 'val'):
            self._save_best_pooled_head(ctx, selector_metrics)

        # ── Round advancement ─────────────────────────────────────────
        if ctx.cur_mode in ('train', 'finetune'):
            self.local_train_round += 1

            # Phase1 -> Phase2 transition
            next_phase = self._decide_phase(
                self.local_train_round, ctx.cur_mode)
            if self.current_phase == 1 and next_phase == 2:
                logger.info(
                    "[FedCQD] Phase 1 complete - running adv head test "
                    "on held-out split.")
                pooled = getattr(ctx.cfg.data, 'pool_prepared_train', False)
                if not pooled:
                    self._run_adv_test(ctx)

                # Notify selector about phase transition
                self.selector.on_phase2_start()

                # Save pretrained adversarial head for future reuse
                if not pooled:
                    self._save_adv_pretrained(ctx)

            # Advance Phase2 warmup counter
            if self.current_phase == 2:
                self.selector.advance_phase2_counter()

    # =====================================================================
    # MAP-Elites: BC computation + dynamic lambda setting
    # =====================================================================

    def set_grl_lambdas(self, lambda_config):
        """
        Set GRL lambda for this round (called by MAPElitesClient
        after receiving the personalized (model, lambda_config) from server).

        lambda_config: {"bias_length": float, "bias_lead": float}
        """
        if not hasattr(self, 'selector') or self.selector is None:
            logger.warning(
                "[FedCQD][set_grl_lambdas] No selector, skip")
            return

        # Update per-factor lambdas
        for factor, lam in lambda_config.items():
            if factor in self.selector.grl_lambdas:
                old = self.selector.grl_lambdas[factor]
                self.selector.grl_lambdas[factor] = float(lam)
                if old != float(lam):
                    logger.info(
                        "[FedCQD][set_grl_lambdas] %s: %.3f -> %.3f",
                        factor, old, float(lam))

        # Update max lambda (used for GRL skip-when-zero optimization)
        self.selector.grl_lambda = (
            max(self.selector.grl_lambdas.values())
            if self.selector.grl_lambdas else 0.0
        )

    def compute_bc(self, max_batches=8):
        """
        Compute Behavior Characteristic (BC) for MAP-Elites.

        Uses the (frozen Phase-1) adversarial selector to forward a
        few batches of local training data, and returns the average
        adv_acc for each bias factor.

        Strategy: reuse selector.compute_loss(use_grl=False) which already
        returns per-factor adv_acc in metrics dict. We just aggregate across
        batches.

        Returns:
            dict with keys:
                adv_len_acc: float   (avg accuracy on bias_length head)
                adv_lead_acc: float  (avg accuracy on bias_lead head)
                val_acc: 0.0         (placeholder, filled by client)
        """
        me_cfg = getattr(self._cfg, 'map_elites', None)
        descriptor_mode = getattr(
            me_cfg, 'descriptor_mode', 'adversarial_accuracy')
        if descriptor_mode == 'data_preference':
            return self._compute_data_preference_bc()

        if not hasattr(self, 'selector') or self.selector is None:
            return {"adv_len_acc": 0.0, "adv_lead_acc": 0.0, "val_acc": 0.0}

        if not self.adv_factors:
            return {"adv_len_acc": 0.0, "adv_lead_acc": 0.0, "val_acc": 0.0}

        import torch

        # device is on self.ctx, not self
        ctx = getattr(self, 'ctx', None)
        if ctx is None:
            logger.warning(
                "[FedCQD][compute_bc] No ctx available, BC=0")
            return {"adv_len_acc": 0.0, "adv_lead_acc": 0.0, "val_acc": 0.0}
        device = ctx.device

        # We need ctx for _encode_pair. Already have it above.

        # Get dataloader from ctx
        loader = getattr(ctx, 'train_loader', None)
        if loader is None:
            logger.warning(
                "[FedCQD][compute_bc] No train_loader on ctx, BC=0")
            return {"adv_len_acc": 0.0, "adv_lead_acc": 0.0, "val_acc": 0.0}

        # Save RNG state before compute_bc to prevent side-effects
        # on subsequent training (standalone mode shares RNG across clients)
        rng_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

        # Switch to eval mode (no dropout, no BN update)
        # NOTE: model lives on ctx.model, not self.model
        ctx.model.eval()
        self.selector.eval()

        # Accumulate per-factor accuracy weighted by batch size
        factor_correct = {f: 0 for f in self.adv_factors}
        factor_total = {f: 0 for f in self.adv_factors}
        n_batches = 0

        try:
            with torch.no_grad():
                for batch in loader:
                    if n_batches >= max_batches:
                        break

                    # Extract win/lose ids from batch
                    win_ids = batch.get('win_input_ids')
                    win_mask = batch.get('win_attention_mask')
                    lose_ids = batch.get('lose_input_ids')
                    lose_mask = batch.get('lose_attention_mask')

                    if win_ids is None or lose_ids is None:
                        continue

                    win_ids = win_ids.to(device)
                    win_mask = win_mask.to(device)
                    lose_ids = lose_ids.to(device)
                    lose_mask = lose_mask.to(device)
                    B = win_ids.size(0)

                    # Check batch has all adv factor labels
                    has_adv = all(
                        f in batch for f in self.adv_factors
                    )
                    if not has_adv:
                        continue

                    # Encode win/lose through LLM backbone (no grad)
                    win_h, lose_h = self._encode_pair(
                        ctx, win_ids, win_mask, lose_ids, lose_mask,
                        require_grad=False, win_labels=batch.get('win_labels'),
                        lose_labels=batch.get('lose_labels')
                    )

                    # Run selector forward (use_grl=False, Phase-1 style)
                    # Returns (loss, metrics) where metrics has adv_{f}_acc
                    try:
                        _, metrics = self.selector.compute_loss(
                            win_h, lose_h, batch,
                            use_grl=False, device=device
                        )
                    except Exception as e:
                        logger.warning(
                            "[FedCQD][compute_bc] selector forward "
                            "failed: %s", e)
                        continue

                    # Accumulate weighted by batch size
                    for factor in self.adv_factors:
                        acc_key = f'adv_{factor}_acc'
                        if acc_key in metrics:
                            factor_correct[factor] += (
                                metrics[acc_key] * B
                            )
                            factor_total[factor] += B

                    n_batches += 1

        except Exception as e:
            logger.warning(
                "[FedCQD][compute_bc] iteration failed: %s", e)
        finally:
            ctx.model.train()
            if hasattr(self.selector, 'train'):
                self.selector.train()
            # Restore RNG state to prevent compute_bc from perturbing
            # the random sequence seen by subsequent client training
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])

        # Compute average accuracy per factor
        bc_info = {"val_acc": 0.0}
        for factor in self.adv_factors:
            total = factor_total[factor]
            acc = factor_correct[factor] / total if total > 0 else 0.0
            if factor == 'bias_length':
                bc_info['adv_len_acc'] = acc
            elif factor == 'bias_lead':
                bc_info['adv_lead_acc'] = acc
            else:
                bc_info[f'adv_{factor}_acc'] = acc

        bc_info.setdefault('adv_len_acc', 0.0)
        bc_info.setdefault('adv_lead_acc', 0.0)

        logger.info(
            "[FedCQD][compute_bc] BC=%s (batches=%d)",
            {k: round(v, 4) for k, v in bc_info.items()},
            n_batches,
        )

        return bc_info

    def _compute_data_preference_bc(self):
        """Compute a stable worker profile directly from preference pairs."""
        from federatedscope.core.auxiliaries.ReIterator import iter_loader_once
        from federatedscope.llm.map_elites_artifacts import \
            compute_data_preference_bc_from_batches

        loader = self._get_bc_dataloader()
        if loader is None:
            raise RuntimeError(
                'data_preference BD requires a local training dataloader')

        me_cfg = self._cfg.map_elites
        factors = list(getattr(
            me_cfg, 'factors', ['bias_length', 'bias_lead']))
        configured_priors = getattr(me_cfg, 'factor_priors', {})
        priors = {
            factor: float(configured_priors.get(
                factor,
                getattr(me_cfg, f'{factor[len("bias_"):]}_prior', 0.0)))
            for factor in factors
        }
        bc_info = compute_data_preference_bc_from_batches(
            batches=iter_loader_once(loader),
            factors=factors,
            shrinkage=float(me_cfg.descriptor_shrinkage),
            priors=priors,
        )
        logger.info(
            '[FedCQD][compute_bc] data-preference BD=%s',
            {key: round(value, 6) if isinstance(value, float) else value
             for key, value in bc_info.items()})
        return bc_info

    def _get_bc_dataloader(self):
        """Get a dataloader for BC computation (reuses training data)."""
        ctx = getattr(self, 'ctx', None)
        if ctx is not None:
            loader = getattr(ctx, 'train_loader', None)
            if loader is not None:
                return loader

        if self.data is not None:
            train_data = getattr(self.data, 'train_data', None) \
                or getattr(self.data, 'train', None)
            if train_data is not None:
                from torch.utils.data import DataLoader
                bs = self._cfg.dataloader.batch_size
                return DataLoader(train_data, batch_size=bs, shuffle=False)

        return None

    def _prepare_bc_batch(self, batch, device):
        """Deprecated: kept for backward compat. New compute_bc does not
        use this — it calls selector.compute_loss directly with win/lose
        pairs which is the correct interface."""
        return None


# =====================================================================
# Registration
# =====================================================================

def call_fedcqd_trainer(trainer_type):
    if trainer_type == 'fedcqd_trainer':
        return FedCQDTrainer


register_trainer('fedcqd_trainer', call_fedcqd_trainer)
