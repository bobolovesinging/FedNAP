"""Artifact helpers for MAP-Elites selector experiments."""

import copy
import random


def _cell_key(cell_id):
    return f'{int(cell_id[0])},{int(cell_id[1])}'


def build_static_balanced_assignments(client_ids, cell_ids, seed):
    """Assign each client to one cell once with balanced group sizes."""
    clients = [int(client_id) for client_id in client_ids]
    if len(set(clients)) != len(clients):
        raise ValueError('duplicate client ids are not allowed')

    cells = [tuple(int(value) for value in cell_id) for cell_id in cell_ids]
    if not cells:
        raise ValueError('at least one cell is required')
    if len(set(cells)) != len(cells):
        raise ValueError('duplicate cell ids are not allowed')
    if len(clients) < len(cells):
        raise ValueError('fewer clients than cells cannot form all groups')

    shuffled_clients = sorted(clients)
    random.Random(int(seed)).shuffle(shuffled_clients)
    sorted_cells = sorted(cells)
    return {
        client_id: sorted_cells[index % len(sorted_cells)]
        for index, client_id in enumerate(shuffled_clients)
    }


def _preference_descriptor_key(factor):
    if not str(factor).startswith('bias_'):
        raise ValueError(f'invalid nuisance factor name: {factor}')
    return f"bd_{str(factor)[len('bias_'):]}_preference"


def compute_data_preference_bc_from_batches(batches, factors, shrinkage,
                                            priors):
    """Aggregate configured data-preference descriptors from worker batches."""
    factors = list(factors)
    if len(factors) != 2 or len(set(factors)) != 2:
        raise ValueError('exactly two distinct nuisance factors are required')
    missing_priors = [factor for factor in factors if factor not in priors]
    if missing_priors:
        raise ValueError(f'missing descriptor priors: {missing_priors}')

    descriptor_keys = {
        factor: _preference_descriptor_key(factor) for factor in factors
    }
    sums = {factor: 0.0 for factor in factors}
    count = 0
    for batch in batches:
        values_by_factor = {}
        for factor, key in descriptor_keys.items():
            values = batch.get(key)
            if values is None:
                raise RuntimeError(
                    f'Worker samples are missing data-preference field {key}')
            values_by_factor[factor] = values.detach().float().reshape(-1)

        batch_counts = {
            int(values.numel()) for values in values_by_factor.values()
        }
        if len(batch_counts) != 1:
            raise RuntimeError('BD field lengths do not match')
        batch_count = batch_counts.pop()
        for factor, values in values_by_factor.items():
            sums[factor] += float(values.sum().item())
        count += batch_count

    if count == 0:
        raise RuntimeError('Cannot compute data-preference BD on empty data')

    shrinkage = float(shrinkage)
    denominator = count + shrinkage
    if denominator <= 0:
        raise ValueError('descriptor shrinkage yields a non-positive denominator')
    result = {
        descriptor_keys[factor]: (
            sums[factor] + shrinkage * float(priors[factor])) / denominator
        for factor in factors
    }
    result.update({'bd_count': count, 'val_acc': 0.0})
    return result


def update_grl_lambdas(selector, lambda_config):
    """Apply one cell's configured nuisance strengths to a selector."""
    for factor, value in lambda_config.items():
        if factor in selector.grl_lambdas:
            selector.grl_lambdas[factor] = float(value)
    selector.grl_lambda = (
        max(selector.grl_lambdas.values())
        if selector.grl_lambdas else 0.0)


def compute_data_preference_client_bc(dataset, expected_client_num,
                                      shrinkage, length_prior=0.0,
                                      lead_prior=0.0, factors=None,
                                      priors=None):
    """Precompute static client BCs using the augmented choice population."""
    factors = list(factors or ['bias_length', 'bias_lead'])
    if len(factors) != 2 or len(set(factors)) != 2:
        raise ValueError('exactly two distinct nuisance factors are required')
    descriptor_keys = {
        factor: _preference_descriptor_key(factor) for factor in factors
    }
    if priors is None:
        priors = {
            'bias_length': float(length_prior),
            'bias_lead': float(lead_prior),
        }
    missing_priors = [factor for factor in factors if factor not in priors]
    if missing_priors:
        raise ValueError(f'missing descriptor priors: {missing_priors}')
    categories = list(getattr(dataset, 'categories', []))
    pair_dataset = getattr(dataset, 'pair_dataset', None)
    pair_labels = getattr(pair_dataset, 'bias_labels', None)
    if not categories or not pair_labels:
        raise ValueError(
            'causal choice dataset must expose categories and pair bias labels')

    pair_count = len(pair_labels)
    grouped = {}
    for index, category in enumerate(categories):
        label = pair_labels[index % pair_count]
        try:
            values = {
                factor: float(label[key])
                for factor, key in descriptor_keys.items()
            }
        except KeyError as error:
            raise ValueError(
                f'missing data-preference descriptor: {error}') from error
        stats = grouped.setdefault(int(category), {
            'sums': {factor: 0.0 for factor in factors},
            'count': 0,
        })
        for factor, value in values.items():
            stats['sums'][factor] += value
        stats['count'] += 1

    expected_categories = set(range(int(expected_client_num)))
    actual_categories = set(grouped)
    if actual_categories != expected_categories:
        raise ValueError(
            'expected client ids '
            f'{sorted(expected_categories)}, observed {sorted(actual_categories)}')

    result = {}
    shrinkage = float(shrinkage)
    for category, stats in grouped.items():
        denominator = stats['count'] + shrinkage
        if denominator <= 0:
            raise ValueError(f'client {category + 1} has no BC samples')
        result[category + 1] = tuple(
            (stats['sums'][factor]
             + shrinkage * float(priors[factor])) / denominator
            for factor in factors
        )
    return result


def evaluate_cell_states(model, trainer, cells, splits):
    """Evaluate each final cell on one server dataset with one model object."""
    original_state = copy.deepcopy(model.state_dict())
    expected_keys = set(original_state)
    results = {}
    try:
        for cell_id in sorted(cells):
            model_state = cells[cell_id].model_state
            if set(model_state) != expected_keys:
                raise ValueError(
                    f'cell {cell_id} state keys do not match server model')
            mismatched = [
                key for key in expected_keys
                if tuple(model_state[key].shape)
                != tuple(original_state[key].shape)
            ]
            if mismatched:
                raise ValueError(
                    f'cell {cell_id} state shapes do not match: {mismatched}')
            model.load_state_dict(model_state, strict=False)
            metrics = {}
            for split in splits:
                metrics.update(trainer.evaluate(
                    target_data_split_name=split))
            results[cell_id] = metrics
    finally:
        model.load_state_dict(original_state, strict=False)
    return results


def build_client_manifest(client_bc, bc_to_cell, cell_ids,
                          expected_client_num, require_all_clients):
    """Build final client-to-cell assignments from each client's latest BC."""
    expected_client_num = int(expected_client_num)
    observed_client_num = len(client_bc)
    if require_all_clients and observed_client_num != expected_client_num:
        raise ValueError(
            f'expected {expected_client_num} client descriptors, '
            f'observed {observed_client_num}')

    valid_cells = {tuple(cell_id) for cell_id in cell_ids}
    counts = {_cell_key(cell_id): 0 for cell_id in sorted(valid_cells)}
    assignments = {}
    for client_id, descriptor in sorted(
            client_bc.items(), key=lambda item: int(item[0])):
        cell_id = tuple(bc_to_cell(*descriptor))
        if cell_id not in valid_cells:
            raise ValueError(
                f'client {client_id} mapped to unknown cell {cell_id}')
        counts[_cell_key(cell_id)] += 1
        assignments[str(client_id)] = {
            'cell_id': list(cell_id),
            'descriptor': [float(descriptor[0]), float(descriptor[1])],
        }

    return {
        'expected_client_num': expected_client_num,
        'observed_client_num': observed_client_num,
        'complete': observed_client_num == expected_client_num,
        'cell_client_counts': counts,
        'client_assignments': assignments,
    }


def build_static_assignment_manifest(client_assignments, cell_ids,
                                     expected_client_num,
                                     require_all_clients, routing_mode,
                                     random_routing_seed):
    """Build a manifest from explicit routing without fabricating BCs."""
    expected_client_num = int(expected_client_num)
    observed_client_num = len(client_assignments)
    if require_all_clients and observed_client_num != expected_client_num:
        raise ValueError(
            f'expected {expected_client_num} client assignments, '
            f'observed {observed_client_num}')

    valid_cells = {tuple(cell_id) for cell_id in cell_ids}
    counts = {_cell_key(cell_id): 0 for cell_id in sorted(valid_cells)}
    assignments = {}
    for client_id, assigned_cell in sorted(
            client_assignments.items(), key=lambda item: int(item[0])):
        cell_id = tuple(assigned_cell)
        if cell_id not in valid_cells:
            raise ValueError(
                f'client {client_id} mapped to unknown cell {cell_id}')
        counts[_cell_key(cell_id)] += 1
        assignments[str(client_id)] = {
            'cell_id': list(cell_id),
            'assignment_source': str(routing_mode),
        }

    return {
        'routing_mode': str(routing_mode),
        'random_routing_seed': int(random_routing_seed),
        'expected_client_num': expected_client_num,
        'observed_client_num': observed_client_num,
        'complete': observed_client_num == expected_client_num,
        'cell_client_counts': counts,
        'client_assignments': assignments,
    }
