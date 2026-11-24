"""
MAP-Elites config for Federated RLHF.

Registers the `map_elites` config section consumed by MAPElitesServer.

YAML usage:
    map_elites:
      grid_dim: [2, 2]
      bc_threshold: 0.5
      bc_recompute_freq: 10
      elite_archive_interval: 5
      lambda_options:
        bias_length: [0.0, 0.5]
        bias_lead: [0.0, 0.5]
      cold_start_lambda:
        bias_length: 0.0
        bias_lead: 0.0
"""
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_map_elites_cfg(cfg):
    cfg.map_elites = CN()
    cfg.map_elites.grid_dim = [2, 2]
    cfg.map_elites.factors = ['bias_length', 'bias_lead']
    cfg.map_elites.bc_threshold = 0.5
    cfg.map_elites.bc_recompute_freq = 10
    cfg.map_elites.routing_mode = 'behavior_descriptor'
    cfg.map_elites.random_routing_seed = 0
    cfg.map_elites.descriptor_mode = 'adversarial_accuracy'
    cfg.map_elites.dynamic_thresholds = True
    cfg.map_elites.length_threshold = 0.5
    cfg.map_elites.lead_threshold = 0.5
    cfg.map_elites.length_prior = 0.0
    cfg.map_elites.lead_prior = 0.0
    cfg.map_elites.descriptor_shrinkage = 0.0
    cfg.map_elites.precompute_static_client_bc = True
    cfg.map_elites.require_all_clients_for_final_manifest = True
    cfg.map_elites.elite_archive_interval = 5

    # Per-dimension lambda options (list, indexed by cell coordinate)
    cfg.map_elites.lambda_options = CN()
    cfg.map_elites.lambda_options.bias_length = [0.0, 0.5]
    cfg.map_elites.lambda_options.bias_lead = [0.0, 0.5]
    cfg.map_elites.lambda_options.bias_first_person = [0.0, 0.5]

    # Cold-start lambda (Round 0 before any BC is known)
    cfg.map_elites.cold_start_lambda = CN()
    cfg.map_elites.cold_start_lambda.bias_length = 0.0
    cfg.map_elites.cold_start_lambda.bias_lead = 0.0
    cfg.map_elites.cold_start_lambda.bias_first_person = 0.0

    cfg.map_elites.factor_thresholds = CN()
    cfg.map_elites.factor_thresholds.bias_length = 0.5
    cfg.map_elites.factor_thresholds.bias_lead = 0.5
    cfg.map_elites.factor_thresholds.bias_first_person = 0.5

    cfg.map_elites.factor_priors = CN()
    cfg.map_elites.factor_priors.bias_length = 0.0
    cfg.map_elites.factor_priors.bias_lead = 0.0
    cfg.map_elites.factor_priors.bias_first_person = 0.0

    cfg.register_cfg_check_fun(assert_map_elites_cfg)


def assert_map_elites_cfg(cfg):
    interval = cfg.map_elites.elite_archive_interval
    assert type(interval) is int and interval > 0, \
        "map_elites.elite_archive_interval must be a positive integer"
    assert cfg.map_elites.routing_mode in [
        'behavior_descriptor', 'random_static_balanced'
    ], "Unsupported map_elites.routing_mode"
    assert cfg.map_elites.descriptor_mode in [
        'adversarial_accuracy', 'data_preference'
    ], "Unsupported map_elites.descriptor_mode"
    assert cfg.map_elites.descriptor_shrinkage >= 0, \
        "map_elites.descriptor_shrinkage must be non-negative"
    factors = list(getattr(
        cfg.map_elites, 'factors', ['bias_length', 'bias_lead']))
    assert len(factors) == 2 and len(set(factors)) == 2, \
        "map_elites.factors must contain two distinct factor names"
    assert len(cfg.map_elites.grid_dim) == len(factors), \
        "map_elites.grid_dim must match map_elites.factors"
    for index, factor in enumerate(factors):
        assert hasattr(cfg.map_elites.lambda_options, factor), \
            f"lambda_options is missing {factor}"
        options = getattr(cfg.map_elites.lambda_options, factor)
        assert cfg.map_elites.grid_dim[index] == len(options), \
            f"grid_dim[{index}] must match lambda_options.{factor}"


register_config("map_elites", extend_map_elites_cfg)
