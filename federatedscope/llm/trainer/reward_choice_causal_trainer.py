import logging
import math
import os

import torch

from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.llm.trainer.fedcqd_trainer import (
    _infer_hidden_size,
    _last_real_token,
    pool_adversarial_hidden,
    create_selector,
)
from federatedscope.llm.trainer.reward_choice_trainer import \
    RewardChoiceTrainer
from federatedscope.register import register_trainer


logger = logging.getLogger(__name__)


def _trainable_gradient_snapshot(model):
    """Clone trainable gradients for diagnostic delta accounting."""
    with torch.no_grad():
        return [parameter.grad.detach().float().clone()
                if parameter.grad is not None else None
                for parameter in model.parameters() if parameter.requires_grad]


def _gradient_component_stats(before, after_choice, after_adversarial):
    """Measure component deltas without changing the training gradients."""
    if not (len(before) == len(after_choice) == len(after_adversarial)):
        raise ValueError('gradient snapshots must contain the same parameters')
    choice_sq = adversarial_sq = combined_sq = dot = 0.0
    for initial, choice_total, final_total in zip(
            before, after_choice, after_adversarial):
        reference = next((v for v in (initial, choice_total, final_total)
                          if v is not None), None)
        if reference is None:
            continue
        initial = torch.zeros_like(reference) if initial is None else initial
        choice_total = (torch.zeros_like(reference)
                        if choice_total is None else choice_total)
        final_total = (torch.zeros_like(reference)
                       if final_total is None else final_total)
        choice_delta = choice_total - initial
        adversarial_delta = final_total - choice_total
        choice_sq += choice_delta.square().sum().item()
        adversarial_sq += adversarial_delta.square().sum().item()
        combined_sq += (final_total - initial).square().sum().item()
        dot += (choice_delta * adversarial_delta).sum().item()
    choice_norm, adversarial_norm = math.sqrt(choice_sq), math.sqrt(adversarial_sq)
    denominator = choice_norm * adversarial_norm
    return {'choice_grad_norm': choice_norm, 'adv_grad_norm': adversarial_norm,
            'adv_choice_grad_ratio': adversarial_norm / choice_norm
            if choice_norm > 0 else float('inf') if adversarial_norm > 0 else 0.,
            'choice_adv_grad_cosine': dot / denominator if denominator > 0 else 0.,
            'combined_grad_norm': math.sqrt(combined_sq)}


class RewardChoiceCausalTrainer(RewardChoiceTrainer):
    """FedBiscuit A/B choice training with frozen GRL bias heads."""

    def __init__(self, model, data, device, config, only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        raw_factors = getattr(config.trainer, 'adv_factors', [])
        self.adv_factors = list(raw_factors) if raw_factors else []
        if not self.adv_factors:
            raise ValueError('causal choice trainer requires adv_factors')
        if getattr(config.trainer, 'adv_heads_frozen', False) is not True:
            raise ValueError('causal choice adversarial heads must be frozen')

        hidden_size = _infer_hidden_size(model, fallback=896)
        selector_type = getattr(
            config.trainer, 'causal_selector_type', 'grl_adversarial')
        self.selector = create_selector(
            selector_type, hidden_size, self.adv_factors, config.trainer)
        selector_device = self.ctx.device
        self.selector.to(selector_device)

        raw_heads = getattr(config.trainer, 'adv_pretrained_heads', None)
        self.adv_pretrained_heads = (
            {str(key): str(value) for key, value in raw_heads.items()}
            if raw_heads is not None else {})
        loaded = self._load_pretrained_heads(selector_device)
        if set(loaded) != set(self.adv_factors):
            missing = sorted(set(self.adv_factors) - set(loaded))
            raise ValueError(
                f'failed to load all frozen adversarial heads: {missing}')

        for parameter in self.selector.parameters():
            parameter.requires_grad = False
        self.selector.eval()
        self.lambda_adv = float(getattr(config.trainer, 'lambda_adv', 1.0))
        self._causal_diag_batches = 0
        logger.info(
            '[ChoiceCausal] official A/B loss + frozen GRL heads; '
            'factors=%s grl_lambdas=%s lambda_adv=%.3f hidden_size=%d',
            self.adv_factors, self.selector.grl_lambdas,
            self.lambda_adv, hidden_size)

    def _load_pretrained_heads(self, device):
        loaded = []
        self.adv_pooling_by_factor = {}
        for factor, path in self.adv_pretrained_heads.items():
            if factor not in self.adv_factors:
                continue
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"adversarial head for {factor!r} not found: {path}")
            checkpoint = torch.load(path, map_location=device)
            pooling = checkpoint.get('adv_pooling', 'last')
            if pooling not in ('last', 'response_mean'):
                raise ValueError(f'Unsupported head pooling: {pooling}')
            if pooling == 'response_mean':
                policy = checkpoint.get('pair_input_policy', {})
                if (not policy.get('require_complete')
                        or not getattr(self._cfg.data, 'require_complete_pair_inputs', False)
                        or self._cfg.llm.tok_len < policy.get('context_limit', 0)):
                    raise ValueError('Response-mean head requires its complete pair input policy')
            self.adv_pooling_by_factor[factor] = pooling
            checkpoint_factors = list(checkpoint.get('adv_factors', []))
            selector_state = checkpoint.get('selector')
            if not isinstance(selector_state, dict):
                raise ValueError(f'checkpoint has no selector state: {path}')
            if factor not in checkpoint_factors:
                raise ValueError(
                    f'{factor!r} not present in checkpoint factors '
                    f'{checkpoint_factors}: {path}')

            source_index = checkpoint_factors.index(factor)
            target_index = self.adv_factors.index(factor)
            source_prefix = f'adv_heads.{source_index}.'
            target_prefix = f'adv_heads.{target_index}.'
            head_state = {
                key.replace(source_prefix, target_prefix, 1): value
                for key, value in selector_state.items()
                if key.startswith(source_prefix)
            }
            if not head_state:
                raise ValueError(
                    f'checkpoint has no parameters for {factor!r}: {path}')
            self.selector.load_state_dict(head_state, strict=False)
            loaded.append(factor)

        logger.info(
            '[ChoiceCausal] Loaded per-factor pretrained heads: %s from %s',
            loaded, self.adv_pretrained_heads)
        return loaded

    @staticmethod
    def _backbone(model):
        backbone = model
        if hasattr(backbone, 'model'):
            backbone = backbone.model
        if hasattr(backbone, 'base_model'):
            backbone = backbone.base_model
        if hasattr(backbone, 'model'):
            backbone = backbone.model
        if hasattr(backbone, 'model') and not isinstance(backbone, tuple):
            backbone = backbone.model
        return backbone

    def _encode_pair(self, ctx):
        win_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
        win_mask = ctx.data_batch['win_attention_mask'].to(ctx.device)
        lose_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
        lose_mask = ctx.data_batch['lose_attention_mask'].to(ctx.device)
        backbone = self._backbone(ctx.model)
        win_output = backbone(input_ids=win_ids, attention_mask=win_mask)
        lose_output = backbone(input_ids=lose_ids, attention_mask=lose_mask)
        modes = getattr(self, 'adv_pooling_by_factor', {})
        if any(mode != 'last' for mode in modes.values()):
            win_views, lose_views = {}, {}
            for factor in self.adv_factors:
                mode = modes.get(factor, 'last')
                win_views[factor] = pool_adversarial_hidden(
                    win_output.last_hidden_state, win_mask,
                    ctx.data_batch.get('win_labels'), mode)
                lose_views[factor] = pool_adversarial_hidden(
                    lose_output.last_hidden_state, lose_mask,
                    ctx.data_batch.get('lose_labels'), mode)
            return win_views, lose_views
        win_hidden = _last_real_token(
            win_output.last_hidden_state, win_mask).float()
        lose_hidden = _last_real_token(
            lose_output.last_hidden_state, lose_mask).float()
        return win_hidden, lose_hidden

    def _backward_component(self, ctx, loss):
        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(loss)
        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(loss)
        else:
            (loss / self.grad_accum_step).backward()

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        self.selector.eval()
        for parameter in self.selector.parameters():
            parameter.requires_grad = False

    def set_grl_lambdas(self, lambda_config):
        """Apply MAP-Elites cell lambdas to the frozen adversarial heads."""
        from federatedscope.llm.map_elites_artifacts import \
            update_grl_lambdas
        update_grl_lambdas(self.selector, lambda_config)

    def compute_bc(self, max_batches=8):
        """Compute the configured data-preference behavior descriptor."""
        descriptor_mode = getattr(
            self._cfg.map_elites, 'descriptor_mode', 'data_preference')
        if descriptor_mode != 'data_preference':
            raise ValueError(
                'RewardChoiceCausalTrainer supports only data_preference BC')
        return self._compute_data_preference_bc()

    def _compute_data_preference_bc(self):
        from federatedscope.core.auxiliaries.ReIterator import ReIterator
        from federatedscope.llm.map_elites_artifacts import \
            compute_data_preference_bc_from_batches

        loader = self._get_bc_dataloader()
        if loader is None:
            raise RuntimeError(
                'data_preference BD requires a local training dataloader')
        finite_loader = loader.loader if isinstance(loader, ReIterator) else loader
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
        return compute_data_preference_bc_from_batches(
            batches=finite_loader,
            factors=factors,
            shrinkage=float(me_cfg.descriptor_shrinkage),
            priors=priors,
        )

    def _get_bc_dataloader(self):
        ctx = getattr(self, 'ctx', None)
        if ctx is not None:
            loader = getattr(ctx, 'train_loader', None)
            if loader is not None:
                return loader
        if self.data is not None:
            train_data = (
                getattr(self.data, 'train_data', None)
                or getattr(self.data, 'train', None))
            if train_data is not None:
                from torch.utils.data import DataLoader
                return DataLoader(
                    train_data,
                    batch_size=self._cfg.dataloader.batch_size,
                    shuffle=False)
        return None

    def _hook_on_batch_forward(self, ctx):
        super()._hook_on_batch_forward(ctx)
        is_training = ctx.cur_mode in (
            MODE.TRAIN, MODE.FINETUNE, 'train', 'finetune')
        if not is_training or ctx.skip_this_batch:
            return

        missing = [factor for factor in self.adv_factors
                   if factor not in ctx.data_batch]
        pair_keys = (
            'win_input_ids', 'win_attention_mask',
            'lose_input_ids', 'lose_attention_mask',
        )
        missing.extend(key for key in pair_keys if key not in ctx.data_batch)
        if missing:
            raise KeyError(
                f'causal choice batch is missing required fields: {missing}')

        diagnostic_batch = self._causal_diag_batches + 1
        log_diagnostics = diagnostic_batch <= 3 or diagnostic_batch % 100 == 0
        gradients_before = (_trainable_gradient_snapshot(ctx.model)
                            if log_diagnostics else None)
        choice_loss = ctx.loss_batch
        if not torch.isfinite(choice_loss):
            raise FloatingPointError('non-finite A/B choice loss')
        choice_loss_value = choice_loss.detach()
        self._backward_component(ctx, choice_loss)
        gradients_after_choice = (_trainable_gradient_snapshot(ctx.model)
                                  if log_diagnostics else None)
        ctx.y_prob = CtxVar(ctx.y_prob.detach(), LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(choice_loss_value, LIFECYCLE.BATCH)
        del choice_loss

        win_hidden, lose_hidden = self._encode_pair(ctx)
        self.selector.eval()
        adversarial_loss, metrics = self.selector.compute_loss(
            win_hidden,
            lose_hidden,
            ctx.data_batch,
            use_grl=True,
            device=ctx.device,
            factor_lambdas=self.selector.grl_lambdas,
        )
        scaled_adversarial_loss = self.lambda_adv * adversarial_loss
        if not torch.isfinite(scaled_adversarial_loss):
            raise FloatingPointError('non-finite frozen-GRL loss')
        self._backward_component(ctx, scaled_adversarial_loss)
        if log_diagnostics:
            gradient_stats = _gradient_component_stats(
                gradients_before, gradients_after_choice,
                _trainable_gradient_snapshot(ctx.model))

        reported_loss = (
            choice_loss_value + scaled_adversarial_loss.detach())
        ctx.loss_batch = CtxVar(reported_loss, LIFECYCLE.BATCH)
        ctx._causal_grads_ready = CtxVar(True, LIFECYCLE.BATCH)

        self._causal_diag_batches = diagnostic_batch
        if log_diagnostics:
            logger.info(
                '[ChoiceCausal] batch=%d choice_loss=%.6f adv_loss=%.6f '
                'choice_grad_norm=%.6f adv_grad_norm=%.6f '
                'grad_ratio=%.6f grad_cosine=%.6f '
                'combined_grad_norm=%.6f adv_metrics=%s',
                self._causal_diag_batches, choice_loss_value.item(),
                adversarial_loss.detach().item(),
                gradient_stats['choice_grad_norm'], gradient_stats['adv_grad_norm'],
                gradient_stats['adv_choice_grad_ratio'],
                gradient_stats['choice_adv_grad_cosine'],
                gradient_stats['combined_grad_norm'], metrics)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return
        if not getattr(ctx, '_causal_grads_ready', False):
            return super()._hook_on_batch_backward(ctx)

        if ctx.cfg.llm.accelerator.use:
            ctx.optimizer.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
            ctx.optimizer.zero_grad()
        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
        elif (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    ctx.model.parameters(), ctx.grad_clip)
            ctx.optimizer.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
            ctx.optimizer.zero_grad()


def call_reward_choice_causal_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicecausaltrainer':
        return RewardChoiceCausalTrainer


register_trainer(
    'llmrewardchoicecausaltrainer', call_reward_choice_causal_trainer)
