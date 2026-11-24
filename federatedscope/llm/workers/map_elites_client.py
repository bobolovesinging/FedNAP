"""
MAP-Elites Client for Federated RLHF

与标准 Client 的差异:
  1. 接收服务器下发的 (model_para, lambda_config) 二元组
  2. 训练前调用 trainer.set_grl_lambdas(lambda_config) 设置本轮 λ
  3. 训练后调用 trainer.compute_bc() 算 BC（Phase 1 对抗头 forward）
  4. 回传 (sample_size, model_para, bc_info) 三元组
     bc_info = {"adv_len_acc", "adv_lead_acc"}
"""
import copy
import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client

logger = logging.getLogger(__name__)


class MAPElitesClient(Client):
    """MAP-Elites client with cached behavior descriptors."""

    def _get_bc_info(self):
        me_cfg = self._cfg.map_elites
        descriptor_mode = getattr(
            me_cfg, 'descriptor_mode', 'adversarial_accuracy')
        static_descriptor = descriptor_mode == 'data_preference'
        recompute_freq = max(int(me_cfg.bc_recompute_freq), 1)
        cached_round = getattr(self, '_map_elites_bc_round', None)
        should_recompute = (
            not hasattr(self, '_map_elites_bc_info')
            or (not static_descriptor
                and (cached_round is None
                     or self.state - cached_round >= recompute_freq)))
        if should_recompute:
            self._map_elites_bc_info = self.trainer.compute_bc()
            self._map_elites_bc_round = self.state
        else:
            logger.info(
                '[MAP-Elites] Client #%s reuses BD from round %s',
                self.ID, self._map_elites_bc_round)
        return copy.deepcopy(self._map_elites_bc_info)

        """
    MAP-Elites 客户端

    核心改动:
      - callback_funcs_for_model_para: 解析 (model_para, lambda_config) 二元组
      - 训练前 set_grl_lambdas, 训练后 compute_bc
      - 回传增加 bc_info
    """

    def callback_funcs_for_model_para(self, message: Message):
        """
        处理服务器下发的 (model_para, lambda_config)

        content 格式: (model_para, lambda_config)
          model_para: state_dict
          lambda_config: {"bias_length": float, "bias_lead": float}
        """
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # ── 解析个性化 content ──
        if isinstance(content, tuple) and len(content) == 2 \
                and isinstance(content[1], dict) \
                and 'bias_length' in content[1]:
            # MAP-Elites 下发: (model_para, lambda_config)
            model_para, lambda_config = content
            is_map_elites_msg = True
        else:
            # 兼容标准格式: model_para
            model_para = content
            lambda_config = None
            is_map_elites_msg = False

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(model_para, list):
                model_para = [
                    symmetric_uniform_dequantization(x) for x in model_para
                ]
            else:
                model_para = symmetric_uniform_dequantization(model_para)

        # 加载模型参数
        if self._cfg.federate.process_num > 1:
            for k, v in model_para.items():
                model_para[k] = v.to(self.device)
        self.trainer.update(
            model_para,
            strict=self._cfg.federate.share_local_model
        )
        self.state = round

        # ── 设置本轮 GRL λ（MAP-Elites 核心） ──
        if lambda_config is not None and hasattr(
                self.trainer, 'set_grl_lambdas'):
            self.trainer.set_grl_lambdas(lambda_config)
            logger.info(
                f"[MAP-Elites] Client #{self.ID} Round {round} "
                f"λ={lambda_config}")

        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]

        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been "
                    f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()
            sample_size, model_para_all, results = self.trainer.train()

            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)

            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)

        # ── 计算 BC（MAP-Elites 核心） ──
        bc_info = {"adv_len_acc": 0.0, "adv_lead_acc": 0.0}
        if hasattr(self.trainer, 'compute_bc'):
            bc_info = self._get_bc_info()

        # ── 回传服务器: (sample_size, model_para, bc_info) ──
        if self._cfg.federate.use_ss:
            # Secret sharing 路径（暂不支持 MAP-Elites，回退到标准格式）
            logger.warning(
                "[MAP-Elites] Secret sharing not supported, "
                "falling back to standard format")
            super().callback_funcs_for_model_para(message)
            return

        # 标准/异步路径
        if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                ['krum', 'normbounding', 'median', 'trimmedmean', 'bulyan']:
            shared_model_para = self._calculate_model_delta(
                init_model=model_para, updated_model=model_para_all)
        else:
            shared_model_para = model_para_all

        # quantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if isinstance(shared_model_para, list):
                shared_model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in shared_model_para
                ]
            else:
                shared_model_para = symmetric_uniform_quantization(
                    shared_model_para, nbits)

        # 回传三元组: (sample_size, model_para, bc_info)
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, shared_model_para, bc_info)))

    def callback_funcs_for_evaluate(self, message: Message):
        """
        Override: handle MAP-Elites content tuple during eval.

        Server broadcasts (model_para, lambda_config) for eval too,
        but standard update() expects only model_para.
        """
        # Strip lambda_config if present
        if isinstance(message.content, tuple) and len(message.content) == 2 \
                and isinstance(message.content[1], dict) \
                and 'bias_length' in message.content[1]:
            model_para, lambda_config = message.content
            # Build a new message-like with just model_para
            # Re-dispatch with stripped content
            from federatedscope.core.message import Message as _Msg
            new_msg = _Msg(
                msg_type=message.msg_type,
                sender=message.sender,
                receiver=message.receiver,
                state=message.state,
                timestamp=message.timestamp,
                content=model_para,
            )
            super().callback_funcs_for_evaluate(new_msg)
        else:
            super().callback_funcs_for_evaluate(message)
