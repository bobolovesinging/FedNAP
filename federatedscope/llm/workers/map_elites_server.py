"""
MAP-Elites Server for Federated RLHF

维护 2x2 BC (Behavior Characteristic) 网格：
  - 维度1: adv_length_acc (<0.5 / >=0.5)
  - 维度2: adv_lead_acc   (<0.5 / >=0.5)
每格独立维护: model + lambda_config + fitness (best val_acc)

每轮流程:
  1. 收集客户端回传 (sample_size, model_para, bc_info)
  2. 按 BC 分组到 4 格
  3. 每格独立 FedAvg 聚合
  4. 按配置在服务器验证集上比较当前模型与档案中的精英
  5. 若验证准确率严格提高，更新精英档案
  6. 下发: 每客户端按其 BC 对应格子的 (model, lambda_config)

Round 0 冷启动: 统一下发 (初始模型, λ={0,0})，客户端训练后回传 BC
"""
import copy
import json
import logging
import os
import random
import numpy as np
import torch

from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import (
    merge_dict_of_results,
    merge_param_dict,
)
from federatedscope.llm.map_elites_artifacts import (
    build_client_manifest,
    build_static_assignment_manifest,
    build_static_balanced_assignments,
    compute_data_preference_client_bc,
    evaluate_cell_states,
)

logger = logging.getLogger(__name__)


class EliteCell:
    """MAP-Elites 网格的一个格子"""

    def __init__(self, cell_id, lambda_config, initial_model_state):
        self.cell_id = cell_id  # (len_idx, lead_idx), e.g. (0,1)
        self.lambda_config = lambda_config  # {"bias_length": 0.0, "bias_lead": 0.5}
        self.model_state = copy.deepcopy(initial_model_state)  # 当前聚合模型
        self.best_model_state = copy.deepcopy(initial_model_state)  # 历史最优
        self.fitness = -1.0  # best server validation accuracy
        self.elite_round = None
        self.n_clients_last = 0  # 上轮落入此格的客户端数
        self.n_updates = 0  # 历史更新次数

    def maybe_update_fitness(self, avg_val_acc, current_model_state,
                             completed_round=None):
        """Track the best candidate without interrupting cumulative training."""
        if avg_val_acc > self.fitness:
            self.fitness = avg_val_acc
            self.best_model_state = copy.deepcopy(current_model_state)
            self.elite_round = completed_round
            return True
        return False

    def __repr__(self):
        return (f"EliteCell{self.cell_id} λ={self.lambda_config} "
                f"fitness={self.fitness:.4f} n_clients={self.n_clients_last} "
                f"n_updates={self.n_updates}")


class MAPElitesServer(Server):
    """
    MAP-Elites 联邦服务器

    配置 (cfg.map_elites):
        grid_dim: [2, 2]                # 网格维度
        bc_threshold: 0.5               # BC 二值化阈值
        lambda_options:                 # 每维 λ 选项
            bias_length: [0.0, 0.5]
            bias_lead: [0.0, 0.5]
        bc_recompute_freq: 10           # BC 重算频率（轮）
        cold_start_lambda:              # Round 0 冷启动 λ
            bias_length: 0.0
            bias_lead: 0.0
    """

    def __init__(self, ID=-1, state=0, config=None, data=None, model=None,
                 client_num=5, total_round_num=10, device='cpu',
                 strategy=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num,
                         total_round_num, device, strategy, **kwargs)

        # ── MAP-Elites 网格初始化 ──────────────────────────
        me_cfg = self._cfg.map_elites
        self.elite_archive_interval = me_cfg.elite_archive_interval
        self.bc_threshold = me_cfg.bc_threshold
        self.bc_recompute_freq = me_cfg.bc_recompute_freq
        self.descriptor_mode = getattr(
            me_cfg, 'descriptor_mode', 'adversarial_accuracy')
        self.routing_mode = getattr(
            me_cfg, 'routing_mode', 'behavior_descriptor')
        self.random_routing_seed = int(getattr(
            me_cfg, 'random_routing_seed', 0))
        self.dynamic_thresholds = bool(getattr(
            me_cfg, 'dynamic_thresholds', True))
        self.factors = list(getattr(
            me_cfg, 'factors', ['bias_length', 'bias_lead']))
        if len(self.factors) != 2 or len(set(self.factors)) != 2:
            raise ValueError(
                'MAP-Elites requires two distinct nuisance factors')
        self.length_threshold = float(getattr(
            me_cfg, 'length_threshold', self.bc_threshold))
        self.lead_threshold = float(getattr(
            me_cfg, 'lead_threshold', self.bc_threshold))
        legacy_thresholds = {
            'bias_length': self.length_threshold,
            'bias_lead': self.lead_threshold,
        }
        configured_thresholds = getattr(
            me_cfg, 'factor_thresholds', None)
        self.factor_thresholds = {
            factor: float(getattr(
                configured_thresholds, factor,
                legacy_thresholds.get(factor, self.bc_threshold)))
            for factor in self.factors
        }
        self.cold_start_lambda = {
            factor: float(getattr(me_cfg.cold_start_lambda, factor))
            for factor in self.factors
        }
        self.factor_lambdas = {
            factor: list(getattr(me_cfg.lambda_options, factor))
            for factor in self.factors
        }
        self.grid_dim = list(me_cfg.grid_dim)

        # 初始模型 state_dict（所有格子共用同一个起点）
        initial_state = copy.deepcopy(self.models[0].state_dict())

        # 构建 2x2 网格
        self.grid = {}
        first_factor, second_factor = self.factors
        for first_index, first_lambda in enumerate(
                self.factor_lambdas[first_factor]):
            for second_index, second_lambda in enumerate(
                    self.factor_lambdas[second_factor]):
                cell_id = (first_index, second_index)
                lam_cfg = {
                    first_factor: float(first_lambda),
                    second_factor: float(second_lambda),
                }
                self.grid[cell_id] = EliteCell(cell_id, lam_cfg, initial_state)

        # 客户端 BC 缓存: {client_id: (adv_len_acc, adv_lead_acc)}
        self.client_bc = {}
        self.cell_server_metrics = {}
        self.dynamic_factor_thresholds = dict(self.factor_thresholds)
        self.dynamic_len_threshold = self.dynamic_factor_thresholds.get(
            'bias_length', self.length_threshold)
        self.dynamic_lead_threshold = self.dynamic_factor_thresholds.get(
            'bias_lead', self.lead_threshold)
        self.client_cell_assignments = {}

        if self.routing_mode == 'random_static_balanced':
            self.client_cell_assignments = build_static_balanced_assignments(
                client_ids=range(1, self._cfg.federate.client_num + 1),
                cell_ids=self.grid.keys(),
                seed=self.random_routing_seed,
            )
            distribution = {cell_id: 0 for cell_id in self.grid}
            for cell_id in self.client_cell_assignments.values():
                distribution[cell_id] += 1
            logger.info(
                '[MAP-Elites] Static random routing seed=%d: %s',
                self.random_routing_seed, distribution)

        if self.routing_mode == 'behavior_descriptor' \
                and self.descriptor_mode == 'data_preference' \
                and bool(getattr(
                me_cfg, 'precompute_static_client_bc', True)):
            train_data = getattr(self.data, 'train_data', None)
            self.client_bc = compute_data_preference_client_bc(
                dataset=train_data,
                expected_client_num=self._cfg.federate.client_num,
                shrinkage=me_cfg.descriptor_shrinkage,
                factors=self.factors,
                priors={
                    factor: float(getattr(
                        getattr(me_cfg, 'factor_priors', None),
                        factor,
                        me_cfg.length_prior if factor == 'bias_length'
                        else me_cfg.lead_prior if factor == 'bias_lead'
                        else 0.0))
                    for factor in self.factors
                },
            )
            distribution = {}
            for descriptor in self.client_bc.values():
                cell_id = self.bc_to_cell(*descriptor)
                distribution[cell_id] = distribution.get(cell_id, 0) + 1
            logger.info(
                '[MAP-Elites] Precomputed static BCs for %d clients: %s',
                len(self.client_bc), distribution)

        # 动态中位数阈值（每轮根据采样客户端 BC 重新计算）
        # 初始用配置的固定阈值，聚合后更新
        logger.info(
            "[MAP-Elites] Grid initialized: %s factors=%s",
            self.grid_dim, self.factors)
        for cell_id, cell in self.grid.items():
            logger.info(f"  Cell {cell_id}: {cell}")

    # ─────────────────────────────────────────────────────
    # BC → 格子映射（使用动态中位数阈值）
    # ─────────────────────────────────────────────────────
    def bc_to_cell(self, first_value, second_value):
        """将 BC 映射到格子坐标（使用动态中位数阈值）
        坐标被 clamp 到 [0, grid_dim-1] 范围内。
        """
        values = (float(first_value), float(second_value))
        coordinates = []
        for index, (factor, value) in enumerate(zip(self.factors, values)):
            threshold = self.dynamic_factor_thresholds[factor]
            coordinate = 0 if value < threshold else 1
            coordinates.append(min(coordinate, self.grid_dim[index] - 1))
        # Clamp to valid grid range (supports 1x1, 2x2, 3x3, etc.)
        return tuple(coordinates)

    def _extract_descriptor(self, bc_info):
        if self.descriptor_mode == 'data_preference':
            required = tuple(
                f"bd_{factor[len('bias_'):]}_preference"
                for factor in self.factors
            )
            missing = [key for key in required if key not in bc_info]
            if missing:
                raise ValueError(
                    f'Missing data-preference descriptors: {missing}')
            return (float(bc_info[required[0]]),
                    float(bc_info[required[1]]))
        legacy_keys = {
            'bias_length': 'adv_len_acc',
            'bias_lead': 'adv_lead_acc',
        }
        keys = [
            legacy_keys.get(factor, f'adv_{factor}_acc')
            for factor in self.factors
        ]
        return tuple(float(bc_info.get(key, 0.0)) for key in keys)

    def _cell_for_client(self, client_id, descriptor=None):
        """Resolve one client's cell using the configured routing policy."""
        client_id = int(client_id)
        if self.routing_mode == 'random_static_balanced':
            try:
                return self.client_cell_assignments[client_id]
            except KeyError as error:
                raise ValueError(
                    f'client {client_id} has no static cell assignment') \
                    from error

        if descriptor is None:
            descriptor = self.client_bc.get(client_id)
        if descriptor is None:
            return None
        return self.bc_to_cell(*descriptor)

    # ─────────────────────────────────────────────────────
    # 重写: 聚合（按 BC 分组）
    # ─────────────────────────────────────────────────────
    def _perform_federated_aggregation(self):
        """
        按 BC 分组聚合: 同格子客户端 FedAvg，不同格子独立维护模型
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]

        # 1. 收集本轮回传，提取 BC 和 model_para
        # msg content 格式: (sample_size, model_para, bc_info)
        # bc_info = {"adv_len_acc": float, "adv_lead_acc": float}
        cell_to_clients = {cell_id: [] for cell_id in self.grid}
        client_to_msg = {}

        for client_id, content in train_msg_buffer.items():
            if len(content) == 3:
                sample_size, model_para, bc_info = content
            else:
                # 兼容旧格式（无 BC）：归到 (0,0) 格
                sample_size, model_para = content
                bc_info = {
                    'adv_len_acc': 0.0, 'adv_lead_acc': 0.0
                }

            # 更新客户端 BC 缓存（每轮都更新，因为客户端每轮都回传）
            descriptor = self._extract_descriptor(bc_info)
            if self.descriptor_mode == 'data_preference' \
                    and client_id in self.client_bc:
                expected = self.client_bc[client_id]
                if any(abs(actual - reference) > 1e-6
                       for actual, reference in zip(descriptor, expected)):
                    raise ValueError(
                        f'Client #{client_id} static BC changed from '
                        f'{expected} to {descriptor}')
            self.client_bc[client_id] = descriptor

            cell_id = self._cell_for_client(client_id, descriptor)
            cell_to_clients[cell_id].append(client_id)
            client_to_msg[client_id] = (sample_size, model_para)

        # 1.5 更新动态中位数阈值（用本轮采样客户端的 BC）
        if self.dynamic_thresholds and self.client_bc:
            for index, factor in enumerate(self.factors):
                values = [bc[index] for bc in self.client_bc.values()]
                if values:
                    self.dynamic_factor_thresholds[factor] = float(
                        np.median(values))
            self.dynamic_len_threshold = self.dynamic_factor_thresholds.get(
                'bias_length', self.dynamic_len_threshold)
            self.dynamic_lead_threshold = self.dynamic_factor_thresholds.get(
                'bias_lead', self.dynamic_lead_threshold)
            logger.info(
                "[MAP-Elites] Dynamic thresholds updated: %s (n=%d)",
                self.dynamic_factor_thresholds,
                len(self.client_bc))

        # 2. 每格独立 FedAvg 聚合
        for cell_id, clients in cell_to_clients.items():
            cell = self.grid[cell_id]
            cell.n_clients_last = len(clients)

            if len(clients) == 0:
                logger.info(
                    f"[MAP-Elites] Round {self.state} Cell {cell_id}: "
                    f"no clients, keep model unchanged")
                continue

            # 收集同格子客户端的 (sample_size, model_para)
            msg_list = [client_to_msg[c] for c in clients]

            # 用 cell 自己的 aggregator 聚合
            # 复用主 aggregator（单模型场景下安全）
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            result = self.aggregators[0].aggregate(agg_info)

            # 更新格子模型
            merged_param = merge_param_dict(
                cell.model_state.copy(), result
            )
            cell.model_state = merged_param
            cell.n_updates += 1

            logger.info(
                f"[MAP-Elites] Round {self.state} Cell {cell_id}: "
                f"clients={clients} aggregated; λ={cell.lambda_config}")

        self._maybe_update_elite_archive()

        # 返回聚合的客户端总数
        aggregated_num = sum(
            len(clients) for clients in cell_to_clients.values()
        )
        return aggregated_num

    def _maybe_update_elite_archive(self):
        """Compare live cells at multiples of the completed-round interval."""
        completed_round = self.state + 1
        if completed_round % self.elite_archive_interval:
            return
        trained_cells = {
            cell_id: cell for cell_id, cell in self.grid.items()
            if cell.n_updates > 0
        }
        # Evaluation in standalone mode shares RNGs with client training.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        devices = (list(range(torch.cuda.device_count()))
                   if torch.cuda.is_initialized() else [])
        try:
            with torch.random.fork_rng(devices=devices):
                self._update_aggregated_cell_fitness(
                    trained_cells, completed_round=completed_round)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def _update_aggregated_cell_fitness(self, updated_cells,
                                        completed_round=None):
        """Evaluate the supplied live cells on the same validation panel."""
        if not updated_cells:
            return {}
        metrics = evaluate_cell_states(
            model=self.models[0],
            trainer=self.trainer,
            cells=updated_cells,
            splits=['val'],
        )
        for cell_id in sorted(updated_cells):
            val_acc = metrics[cell_id].get('val_acc')
            if val_acc is None or not np.isfinite(val_acc):
                raise ValueError(
                    f'Cell {cell_id} returned invalid val_acc: {val_acc}')
            cell = updated_cells[cell_id]
            replaced = cell.maybe_update_fitness(
                float(val_acc), cell.model_state, completed_round)
            logger.info(
                '[MAP-Elites] Round %d Cell %s post-aggregation '
                'val_acc=%.4f fitness=%s%.4f',
                self.state, cell_id, val_acc,
                '*' if replaced else '=', cell.fitness)
        return metrics

    # ─────────────────────────────────────────────────────
    # 重写: 广播（个性化下发模型+λ）
    # ─────────────────────────────────────────────────────
    def broadcast_model_para(self, msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        个性化下发: 每个客户端按其 BC 对应格子的 (model, lambda_config)
        Round 0 冷启动: 统一下发初始模型 + cold_start_lambda
        """
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        skip_broadcast = self._cfg.federate.method in ["local", "global"]

        # ── 为每个接收者准备个性化 content ──
        # content = (model_para, lambda_config)
        # Round 0: 所有客户端用 cold_start_lambda
        # Round >=1: 已知 BC 的客户端按 BC 取格子；未知 BC 的客户端
        #           用"多数格子"（上轮客户端数最多的格子）作为默认，
        #           避免新采样的客户端被错误地退化到 (0,0) 而收到 λ=0。
        is_cold_start = self.routing_mode == 'behavior_descriptor' and (
            (self.state == 0) or (len(self.client_bc) == 0))

        # 计算多数格子（用于未知 BC 客户端的默认）
        if not is_cold_start:
            majority_cell = max(
                self.grid.keys(),
                key=lambda cid: self.grid[cid].n_clients_last
            )
            # 若所有格子都没有客户端（极端情况），退化到 cold_start
            if self.grid[majority_cell].n_clients_last == 0:
                majority_cell = None
        else:
            majority_cell = None

        personalized_contents = []
        for client_id in receiver:
            if self.routing_mode == 'random_static_balanced':
                cell_id = self._cell_for_client(client_id)
                cell = self.grid[cell_id]
                model_para = {} if skip_broadcast else copy.deepcopy(
                    cell.model_state)
                personalized_contents.append(
                    (client_id, model_para, cell.lambda_config))
                continue
            if is_cold_start:
                # 冷启动: 用初始模型 + cold_start_lambda
                model_para = {} if skip_broadcast else copy.deepcopy(
                    self.models[0].state_dict()
                )
                lam_cfg = self.cold_start_lambda
            else:
                # 按客户端 BC 取格子
                bc = self.client_bc.get(client_id, None)
                if bc is None:
                    # 未知 BC（新采样的客户端）: 用多数格子
                    if majority_cell is not None:
                        cell_id = majority_cell
                        logger.info(
                            "[MAP-Elites] Client #%s unknown BC, "
                            "using majority cell %s",
                            client_id, cell_id)
                    else:
                        # 极端情况: 无任何历史 BC，用 cold_start
                        model_para = {} if skip_broadcast else \
                            copy.deepcopy(self.models[0].state_dict())
                        lam_cfg = self.cold_start_lambda
                        personalized_contents.append(
                            (client_id, model_para, lam_cfg))
                        continue
                else:
                    cell_id = self._cell_for_client(client_id, bc)
                cell = self.grid[cell_id]
                model_para = {} if skip_broadcast else copy.deepcopy(
                    cell.model_state
                )
                lam_cfg = cell.lambda_config

            personalized_contents.append((client_id, model_para, lam_cfg))

        # ── 逐客户端发送（因为 content 个性化） ──
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state
        for client_id, model_para, lam_cfg in personalized_contents:
            content = (model_para, lam_cfg)
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[client_id],
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=content))

        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    # ─────────────────────────────────────────────────────
    # 重写: 接收客户端回传（解析 BC 信息）
    # ─────────────────────────────────────────────────────
    def callback_funcs_model_para(self, message: Message):
        """
        处理客户端回传: (sample_size, model_para, bc_info)
        bc_info = {"adv_len_acc", "adv_lead_acc"}
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        assert timestamp >= self.cur_timestamp
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            self.staled_msg_buffer.append((round, sender, content))
        else:
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        move_on_flag = self.check_and_move_on()

        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag

    # ─────────────────────────────────────────────────────
    # 重写: 训练结束时的网格状态报告
    # ─────────────────────────────────────────────────────

    def eval(self):
        """Evaluate every cell on the same server-held val/test panels."""
        if not self._cfg.federate.make_global_eval:
            raise ValueError(
                'MAP-Elites requires federate.make_global_eval=True so '
                'all cells use one aligned server evaluation panel')

        raw_metrics = evaluate_cell_states(
            model=self.models[0],
            trainer=self.trainer,
            cells=self.grid,
            splits=self._cfg.eval.split,
        )
        self.cell_server_metrics = {
            cell_id: {
                key: float(value)
                for key, value in metrics.items()
            }
            for cell_id, metrics in raw_metrics.items()
        }
        for cell_id in sorted(self.cell_server_metrics):
            logger.info(
                '[MAP-Elites] Server eval round=%d cell=%s metrics=%s',
                self.state, cell_id, self.cell_server_metrics[cell_id])

        best_cell_id = max(
            self.cell_server_metrics,
            key=lambda cell_id: (
                self.cell_server_metrics[cell_id].get('val_acc', float('-inf')),
                tuple(-value for value in cell_id),
            ))
        best_metrics = self.cell_server_metrics[best_cell_id]
        formatted_eval_res = self._monitor.format_eval_res(
            best_metrics,
            rnd=self.state,
            role=f'MAP-Elites Best Cell {best_cell_id}',
            forms=self._cfg.eval.report,
            return_raw=True)
        self._monitor.update_best_result(
            self.best_results,
            formatted_eval_res['Results_raw'],
            results_type='map_elites_server_best_cell')
        self.history_results = merge_dict_of_results(
            self.history_results, formatted_eval_res)
        self._monitor.save_formatted_results(formatted_eval_res)
        logger.info(formatted_eval_res)
        self.check_and_save()
    
    # ─────────────────────────────────────────────────────
    # 重写: 训练结束时保存所有 cell 的完整模型（含 score_head + selector）
    # ─────────────────────────────────────────────────────
    def save_final_cell_models(self):
        """保存 4 个 cell 的完整状态，供下游人类偏好学习使用"""
        save_stem = os.path.splitext(self._cfg.federate.save_to)[0]
        save_dir = f'{save_stem}_final_cells'
        missing = [cell_id for cell_id, cell in self.grid.items()
                   if cell.elite_round is None]
        if missing:
            raise ValueError(f'No evaluated elite for cells: {missing}')
        selection_policy = 'periodic_elite'
        os.makedirs(save_dir, exist_ok=False)

        require_all = bool(getattr(
            self._cfg.map_elites,
            'require_all_clients_for_final_manifest',
            True))
        if self.routing_mode == 'random_static_balanced':
            client_manifest = build_static_assignment_manifest(
                client_assignments=self.client_cell_assignments,
                cell_ids=self.grid.keys(),
                expected_client_num=self._cfg.federate.client_num,
                require_all_clients=require_all,
                routing_mode=self.routing_mode,
                random_routing_seed=self.random_routing_seed,
            )
        else:
            client_manifest = build_client_manifest(
                client_bc=self.client_bc,
                bc_to_cell=self.bc_to_cell,
                cell_ids=self.grid.keys(),
                expected_client_num=self._cfg.federate.client_num,
                require_all_clients=require_all,
            )
            client_manifest['routing_mode'] = self.routing_mode
        counts = client_manifest['cell_client_counts']
        client_manifest['selection_state_policy'] = selection_policy
        client_manifest['training_state_policy'] = 'cumulative_live'
        client_manifest['completed_rounds'] = int(self.state)
        cell_records = []
        for cell_id, cell in self.grid.items():
            save_path = os.path.join(save_dir, f'cell_{cell_id[0]}_{cell_id[1]}.pt')
            client_count = counts[f'{cell_id[0]},{cell_id[1]}']
            downstream_state = cell.best_model_state
            selection_metrics = {'val_acc': float(cell.fitness)}
            checkpoint = {
                'cell_id': cell_id,
                'lambda_config': cell.lambda_config,
                'model_state': downstream_state,
                'latest_model_state': cell.model_state,
                'best_model_state': cell.best_model_state,
                'selection_state_policy': selection_policy,
                'completed_rounds': int(self.state),
                'client_val_fitness': cell.fitness,
                'client_count': client_count,
                'n_updates': cell.n_updates,
                'server_eval': selection_metrics,
            }
            record = {
                'cell_id': list(cell_id),
                'lambda_config': cell.lambda_config,
                'client_count': client_count,
                'client_val_fitness': float(cell.fitness),
                'n_updates': int(cell.n_updates),
                'checkpoint': os.path.basename(save_path),
                'selection_state_policy': selection_policy,
                'server_eval': selection_metrics,
            }
            elite_metadata = {
                'checkpoint_source': selection_policy,
                'elite_round': cell.elite_round,
                'elite_val_acc': cell.fitness,
            }
            checkpoint.update(elite_metadata)
            record.update(elite_metadata)
            torch.save(checkpoint, save_path)
            cell_records.append(record)
            logger.info(
                '[MAP-Elites] Saved cell %s (clients=%d, fitness=%.4f) -> %s',
                cell_id, client_count, cell.fitness, save_path)

        client_manifest['cells'] = cell_records
        client_manifest.update({
            'checkpoint_source': selection_policy,
            'completed_round': self.state,
            'elite_archive_interval': self.elite_archive_interval,
        })
        client_manifest['factors'] = list(self.factors)
        client_manifest['descriptor_thresholds'] = {
            factor: float(self.dynamic_factor_thresholds[factor])
            for factor in self.factors
        }
        manifest_path = os.path.join(save_dir, 'manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as file:
            json.dump(client_manifest, file, indent=2, sort_keys=True)
        logger.info('[MAP-Elites] Saved final manifest -> %s', manifest_path)

        # Save the highest-scoring final elite across niches.
        best_cell_id = max(
            self.grid,
            key=lambda cell_id: (
                self.grid[cell_id].fitness,
                tuple(-value for value in cell_id),
            ))
        best_cell = self.grid[best_cell_id]
        best_path = os.path.join(save_dir, 'best_cell.pt')
        torch.save({
            'cell_id': best_cell_id,
            'lambda_config': best_cell.lambda_config,
            'model_state': best_cell.best_model_state,
            'elite_round': best_cell.elite_round,
            'checkpoint_source': selection_policy,
            'latest_model_state': best_cell.model_state,
            'selection_state_policy': selection_policy,
            'completed_rounds': int(self.state),
            'client_val_fitness': best_cell.fitness,
            'server_eval': {'val_acc': float(best_cell.fitness)},
        }, best_path)
        logger.info(f'[MAP-Elites] Best cell {best_cell_id} (fitness={best_cell.fitness:.4f}) -> {best_path}')

    def terminate(self, msg_type='finish'):
        """训练结束时保存所有 cell 模型，再调用父类 terminate"""
        if getattr(self, '_elite_cells_exported', False):
            return
        self.save_final_cell_models()
        self._elite_cells_exported = True
        super().terminate(msg_type=msg_type)


def check_and_save(self):
        """在 eval 后追加 MAP-Elites 网格状态日志"""
        super().check_and_save()

        if self.state % self._cfg.eval.freq == 0:
            logger.info("=" * 60)
            logger.info(f"[MAP-Elites] Grid state at Round {self.state}:")
            for cell_id in sorted(self.grid.keys()):
                cell = self.grid[cell_id]
                logger.info(f"  {cell}")
            # 客户端 BC 分布
            if self.client_bc:
                bc_dist = {}
                for cid, bc in self.client_bc.items():
                    cell_id = self.bc_to_cell(bc[0], bc[1])
                    bc_dist[cell_id] = bc_dist.get(cell_id, 0) + 1
                logger.info(f"[MAP-Elites] Client BC distribution: {bc_dist}")
            logger.info("=" * 60)


def call_map_elites_worker(method):
    """注册函数: 当 cfg.federate.method == 'map_elites' 时触发"""
    if method == 'map_elites':
        from federatedscope.llm.workers.map_elites_client import \
            MAPElitesClient
        return {'server': MAPElitesServer, 'client': MAPElitesClient}
    return None
