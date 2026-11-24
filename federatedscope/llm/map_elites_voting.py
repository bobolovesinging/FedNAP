"""Deterministic four-selector voting for MAP-Elites preference pairs."""

import copy
from collections import Counter


TIE_BREAK_MODES = ('client_count', 'reliability_weighted')


def load_cell_checkpoint(model, checkpoint_path, map_location='cpu', expected_cell=None):
    """Load one cell's trainable state into an existing selector model."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if expected_cell is not None and (
            checkpoint.get('checkpoint_source') != 'periodic_elite'
            or list(checkpoint.get('cell_id', ())) != expected_cell['cell_id']
            or checkpoint.get('elite_round') != expected_cell['elite_round']
            or checkpoint.get('elite_val_acc') != expected_cell['server_val_acc']):
        raise ValueError(f'Elite checkpoint does not match manifest: {checkpoint_path}')
    model_state = checkpoint.get('model_state')
    if not isinstance(model_state, dict):
        raise ValueError(f'checkpoint has no model_state: {checkpoint_path}')

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    checkpoint_keys = set(model_state)
    if checkpoint_keys != expected_keys:
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        raise ValueError(
            'cell state keys do not match the selector trainable state; '
            f'missing={missing}, unexpected={unexpected}')
    shape_mismatches = [
        key for key in expected_keys
        if tuple(expected_state[key].shape) != tuple(model_state[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(
            f'cell state tensor shapes do not match: {shape_mismatches}')

    model.load_state_dict(model_state, strict=False)
    return checkpoint


def predict_with_four_cells(model, checkpoint_paths, predict_fn,
                            map_location='cpu'):
    """Sequentially load four cell states into one model and collect votes."""
    if len(checkpoint_paths) != 4:
        raise ValueError('exactly four cell checkpoints are required')
    votes = []
    for checkpoint_path in checkpoint_paths:
        load_cell_checkpoint(model, checkpoint_path, map_location)
        votes.append(int(predict_fn(model)))
    return votes


def _validate(votes, cells):
    if len(votes) != 4 or len(cells) != 4:
        raise ValueError('four votes and four cell records are required')
    if any(vote not in (0, 1) for vote in votes):
        raise ValueError('selector votes must be binary')
    cell_ids = [tuple(cell['cell_id']) for cell in cells]
    if len(set(cell_ids)) != 4:
        raise ValueError('cell ids must be unique')
    if any(int(cell['client_count']) < 0 for cell in cells):
        raise ValueError('client counts must be non-negative')


def resolve_four_votes(votes, cells, tie_break_mode='client_count'):
    """Resolve four aligned selector votes and return an audit record."""
    votes = [int(vote) for vote in votes]
    _validate(votes, cells)
    if tie_break_mode not in TIE_BREAK_MODES:
        raise ValueError(
            f'unsupported tie_break_mode: {tie_break_mode}; '
            f'expected one of {TIE_BREAK_MODES}')
    vote_counts = {0: votes.count(0), 1: votes.count(1)}
    camp_client_counts = {
        choice: sum(
            int(cell['client_count'])
            for vote, cell in zip(votes, cells)
            if vote == choice)
        for choice in (0, 1)
    }

    if vote_counts[0] != vote_counts[1]:
        winner = 0 if vote_counts[0] > vote_counts[1] else 1
        reason = 'majority'
    else:
        if tie_break_mode == 'reliability_weighted':
            weights = [
                int(cell['client_count']) * max(
                    float(cell['server_val_acc']) - 0.5, 0.0)
                for cell in cells
            ]
            camp_reliability_scores = {
                choice: sum(
                    weight for vote, weight in zip(votes, weights)
                    if vote == choice)
                for choice in (0, 1)
            }
            if camp_reliability_scores[0] != camp_reliability_scores[1]:
                winner = (
                    0 if camp_reliability_scores[0] >
                    camp_reliability_scores[1] else 1)
                reason = 'reliability_weighted'
            else:
                winner = None
                reason = None
        elif camp_client_counts[0] != camp_client_counts[1]:
            winner = (
                0 if camp_client_counts[0] > camp_client_counts[1] else 1)
            reason = 'client_count'
        else:
            winner = None
            reason = None

    if vote_counts[0] == vote_counts[1] and winner is None:
        camp_best_val = {
            choice: max(
                float(cell['server_val_acc'])
                for vote, cell in zip(votes, cells)
                if vote == choice)
            for choice in (0, 1)
        }
        if camp_best_val[0] != camp_best_val[1]:
            winner = 0 if camp_best_val[0] > camp_best_val[1] else 1
            reason = 'server_val_acc'
        else:
            first_index = min(
                range(4), key=lambda index: tuple(cells[index]['cell_id']))
            winner = votes[first_index]
            reason = 'cell_id'

    decision = {
        'winner': winner,
        'reason': reason,
        'vote_counts': vote_counts,
        'camp_client_counts': camp_client_counts,
    }
    if tie_break_mode == 'reliability_weighted' and \
            vote_counts[0] == vote_counts[1]:
        decision['camp_reliability_scores'] = camp_reliability_scores
    return decision


def cells_from_manifest(manifest):
    """Extract the four ordered voting records from a final manifest."""
    if manifest.get('checkpoint_source') != 'periodic_elite':
        raise ValueError('Expected the final validation-selected elite archive')
    interval = manifest['elite_archive_interval']
    if not manifest.get('complete'):
        raise ValueError('MAP-Elites manifest is not complete')
    expected = int(manifest.get('expected_client_num', -1))
    observed = int(manifest.get('observed_client_num', -1))
    if expected != observed:
        raise ValueError(
            f'MAP-Elites client coverage mismatch: {observed}/{expected}')

    cells = []
    for record in manifest.get('cells', []):
        elite_round = record.get('elite_round')
        if (not elite_round or interval < 1 or elite_round % interval
                or elite_round > manifest['completed_rounds']):
            raise ValueError(f"Invalid elite round for niche {record['cell_id']}")
        cells.append({
            'elite_round': elite_round,
            'checkpoint': record['checkpoint'],
            'cell_id': list(record['cell_id']),
            'client_count': int(record['client_count']),
            'server_val_acc': float(record['server_eval']['val_acc']),
        })
    cells.sort(key=lambda cell: tuple(cell['cell_id']))
    _validate([0, 0, 0, 0], cells)
    if sum(cell['client_count'] for cell in cells) != expected:
        raise ValueError('MAP-Elites cell client counts do not sum to coverage')
    return cells


def build_voted_preferences(records, votes_by_cell, cells,
                            tie_break_mode='client_count'):
    """Apply four selector vote columns and return records plus audit summary."""
    if len(votes_by_cell) != 4:
        raise ValueError('exactly four vote columns are required')
    num_records = len(records)
    if any(len(votes) != num_records for votes in votes_by_cell):
        raise ValueError('each vote column must cover every preference record')

    selected = []
    reason_counts = Counter()
    choice_counts = Counter()
    for index, source in enumerate(records):
        votes = [int(cell_votes[index]) for cell_votes in votes_by_cell]
        decision = resolve_four_votes(
            votes, cells, tie_break_mode=tie_break_mode)
        sample = copy.deepcopy(source)
        sample['choice'] = decision['winner']
        sample['map_elites_votes'] = votes
        sample['map_elites_vote_reason'] = decision['reason']
        sample['map_elites_vote_counts'] = {
            str(key): value for key, value in decision['vote_counts'].items()
        }
        sample['map_elites_camp_client_counts'] = {
            str(key): value
            for key, value in decision['camp_client_counts'].items()
        }
        if tie_break_mode != 'client_count':
            sample['map_elites_tie_break_mode'] = tie_break_mode
            if 'camp_reliability_scores' in decision:
                sample['map_elites_camp_reliability_scores'] = {
                    str(key): value
                    for key, value in
                    decision['camp_reliability_scores'].items()
                }
        selected.append(sample)
        reason_counts[decision['reason']] += 1
        choice_counts[decision['winner']] += 1

    cell_vote_counts = []
    for cell, votes in zip(cells, votes_by_cell):
        counts = Counter(int(vote) for vote in votes)
        cell_vote_counts.append({
            'cell_id': list(cell['cell_id']),
            'choice_0': counts[0],
            'choice_1': counts[1],
        })

    reasons = ['majority', 'client_count', 'server_val_acc', 'cell_id']
    if tie_break_mode != 'client_count':
        reasons.insert(2, 'reliability_weighted')
    summary = {
        'num_records': num_records,
        'choice_counts': {
            '0': choice_counts[0],
            '1': choice_counts[1],
        },
        'reason_counts': {
            reason: reason_counts[reason]
            for reason in reasons
        },
        'cell_vote_counts': cell_vote_counts,
    }
    if tie_break_mode != 'client_count':
        summary['tie_break_mode'] = tie_break_mode
    return selected, summary
