"""Downstream improvement helpers for MAP-Elites preference ensembles."""

import copy
import math
import random
import statistics
from collections import Counter

from federatedscope.llm.map_elites_voting import resolve_four_votes


def _validate_columns(records, votes_by_cell, cells):
    if len(votes_by_cell) != 4 or len(cells) != 4:
        raise ValueError("four vote columns and four cells are required")
    if any(len(column) != len(records) for column in votes_by_cell):
        raise ValueError("vote columns must cover every preference record")


def normalized_reliability_confidence(votes, cells):
    """Return normalized camp-score separation for one 2-2 vote."""
    if len(votes) != 4 or len(cells) != 4:
        raise ValueError("four votes and four cells are required")
    if votes.count(0) != 2 or votes.count(1) != 2:
        raise ValueError("reliability confidence requires a 2-2 vote")
    weights = [
        int(cell["client_count"])
        * max(float(cell["server_val_acc"]) - 0.5, 0.0)
        for cell in cells
    ]
    scores = {
        choice: sum(weight for vote, weight in zip(votes, weights)
                    if int(vote) == choice)
        for choice in (0, 1)
    }
    denominator = max(scores[0] + scores[1], 1e-12)
    return abs(scores[0] - scores[1]) / denominator, scores


def filter_agreement_preferences(records, votes_by_cell, cells):
    """Keep all majorities and the upper half of reliability-scored ties."""
    _validate_columns(records, votes_by_cell, cells)
    decisions = []
    tie_confidences = []
    for index in range(len(records)):
        votes = [int(column[index]) for column in votes_by_cell]
        decision = resolve_four_votes(
            votes, cells, tie_break_mode="reliability_weighted")
        confidence = None
        if votes.count(0) == votes.count(1):
            confidence, _ = normalized_reliability_confidence(votes, cells)
            tie_confidences.append(confidence)
        decisions.append((votes, decision, confidence))

    threshold = (statistics.median(tie_confidences)
                 if tie_confidences else None)
    selected = []
    choice_counts = Counter()
    retained_ties = 0
    discarded_ties = 0
    majority_count = 0
    for source, (votes, decision, confidence) in zip(records, decisions):
        is_tie = confidence is not None
        keep = not is_tie or confidence >= threshold
        if is_tie:
            if keep:
                retained_ties += 1
            else:
                discarded_ties += 1
        else:
            majority_count += 1
        if not keep:
            continue
        sample = copy.deepcopy(source)
        sample["choice"] = int(decision["winner"])
        sample["map_elites_votes"] = votes
        sample["map_elites_vote_reason"] = decision["reason"]
        sample["map_elites_filter_rule"] = \
            "upper_half_normalized_reliability"
        if is_tie:
            sample["map_elites_tie_confidence"] = confidence
        selected.append(sample)
        choice_counts[sample["choice"]] += 1

    summary = {
        "filter_rule": "upper_half_normalized_reliability",
        "source_count": len(records),
        "retained_count": len(selected),
        "majority_count": majority_count,
        "tie_count": len(tie_confidences),
        "retained_tie_count": retained_ties,
        "discarded_tie_count": discarded_ties,
        "tie_confidence_threshold": threshold,
        "choice_counts": {
            "0": choice_counts[0],
            "1": choice_counts[1],
        },
    }
    return selected, summary


def _binary_inputs(margins, labels):
    import torch

    if len(margins) != len(labels) or not margins:
        raise ValueError("margins and labels must be non-empty and aligned")
    if any(int(label) not in (0, 1) for label in labels):
        raise ValueError("calibration labels must be binary")
    x = torch.tensor(margins, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    if not torch.isfinite(x).all():
        raise ValueError("calibration margins contain non-finite values")
    return x, y


def fit_temperature(margins, labels):
    """Fit one positive binary-logit temperature by validation NLL."""
    import torch
    import torch.nn.functional as functional

    x, y = _binary_inputs(margins, labels)
    raw_temperature = torch.tensor(
        0.541324854612918, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [raw_temperature], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = functional.softplus(raw_temperature) + 1e-6
        loss = functional.binary_cross_entropy_with_logits(
            x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = functional.softplus(raw_temperature).item() + 1e-6
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise FloatingPointError("temperature calibration is non-finite")
    return float(temperature)


def deterministic_stratified_folds(labels, n_splits=5, seed=0):
    """Build deterministic test-fold indices while preserving both classes."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    groups = {0: [], 1: []}
    for index, label in enumerate(labels):
        label = int(label)
        if label not in groups:
            raise ValueError("stratified labels must be binary")
        groups[label].append(index)
    if any(len(indices) < n_splits for indices in groups.values()):
        raise ValueError("each class must cover every stratified fold")
    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for label in (0, 1):
        indices = list(groups[label])
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_splits].append(index)
    return [sorted(fold) for fold in folds]


def _fit_logistic(features, labels, l2):
    import torch
    import torch.nn.functional as functional

    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("logistic features and labels are not aligned")
    weights = torch.zeros(x.shape[1], dtype=torch.float64,
                          requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias], max_iter=75, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = x @ weights + bias
        loss = functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(l2) * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    values = weights.detach().tolist()
    intercept = float(bias.detach().item())
    if not all(math.isfinite(value) for value in values + [intercept]):
        raise FloatingPointError("logistic calibration is non-finite")
    return values, intercept


def _features(margins_by_cell, temperatures, indices=None):
    length = len(margins_by_cell[0])
    if any(len(column) != length for column in margins_by_cell):
        raise ValueError("all margin columns must have equal length")
    if indices is None:
        indices = range(length)
    return [[
        float(margins_by_cell[cell][index]) / temperatures[cell]
        for cell in range(4)
    ] for index in indices]


def _accuracy(logits, labels):
    choices = [1 if float(logit) >= 0.0 else 0 for logit in logits]
    return sum(choice == int(label)
               for choice, label in zip(choices, labels)) / len(labels)


def confidence_from_logit(logit):
    """Convert a signed binary logit to symmetric confidence in [0, 1)."""
    value = float(logit)
    if not math.isfinite(value):
        raise ValueError("calibrated logit must be finite")
    probability = 1.0 / (1.0 + math.exp(-abs(value)))
    return 2.0 * (probability - 0.5)


def select_calibrated_ensemble(validation_margins, labels, pair_margins,
                               client_counts, l2_grid=None):
    """Select a validation-only soft ensemble and label all pair margins."""
    if len(validation_margins) != 4 or len(pair_margins) != 4:
        raise ValueError("calibration requires four selector margin columns")
    if len(client_counts) != 4 or any(int(count) < 0
                                     for count in client_counts):
        raise ValueError("calibration requires four non-negative counts")
    if any(len(column) != len(labels) for column in validation_margins):
        raise ValueError("validation margins do not align with labels")
    if any(not math.isfinite(float(value))
           for column in validation_margins + pair_margins
           for value in column):
        raise ValueError("ensemble margins contain non-finite values")
    if l2_grid is None:
        l2_grid = [0.001, 0.01, 0.1, 1.0]

    folds = deterministic_stratified_folds(labels, n_splits=5, seed=0)
    soft_scores = []
    stacker_scores = {float(l2): [] for l2 in l2_grid}
    all_indices = set(range(len(labels)))
    for validation_indices in folds:
        train_indices = sorted(all_indices - set(validation_indices))
        train_labels = [int(labels[index]) for index in train_indices]
        fold_labels = [int(labels[index]) for index in validation_indices]
        temperatures = [
            fit_temperature(
                [validation_margins[cell][index]
                 for index in train_indices],
                train_labels,
            )
            for cell in range(4)
        ]
        train_features = _features(
            validation_margins, temperatures, train_indices)
        fold_features = _features(
            validation_margins, temperatures, validation_indices)
        soft_logits = [
            sum(int(count) * value
                for count, value in zip(client_counts, row))
            for row in fold_features
        ]
        soft_scores.append(_accuracy(soft_logits, fold_labels))
        for l2 in stacker_scores:
            weights, bias = _fit_logistic(
                train_features, train_labels, l2)
            logits = [sum(weight * value
                          for weight, value in zip(weights, row)) + bias
                      for row in fold_features]
            stacker_scores[l2].append(_accuracy(logits, fold_labels))

    soft_cv_accuracy = sum(soft_scores) / len(soft_scores)
    stacker_cv_by_l2 = {
        l2: sum(scores) / len(scores)
        for l2, scores in stacker_scores.items()
    }
    best_l2 = min(
        stacker_cv_by_l2,
        key=lambda l2: (-stacker_cv_by_l2[l2], l2),
    )
    stacker_cv_accuracy = stacker_cv_by_l2[best_l2]
    selected_method = (
        "temperature_soft_vote"
        if soft_cv_accuracy >= stacker_cv_accuracy else "l2_stacker")

    temperatures = [
        fit_temperature(column, labels) for column in validation_margins
    ]
    pair_features = _features(pair_margins, temperatures)
    parameters = {}
    if selected_method == "temperature_soft_vote":
        pair_logits = [
            sum(int(count) * value
                for count, value in zip(client_counts, row))
            for row in pair_features
        ]
    else:
        validation_features = _features(validation_margins, temperatures)
        weights, bias = _fit_logistic(
            validation_features, labels, best_l2)
        pair_logits = [sum(weight * value
                           for weight, value in zip(weights, row)) + bias
                       for row in pair_features]
        parameters = {"weights": weights, "bias": bias}
    choices = [1 if logit >= 0.0 else 0 for logit in pair_logits]
    return {
        "selected_method": selected_method,
        "folds": 5,
        "soft_vote_cv_accuracy": soft_cv_accuracy,
        "stacker_cv_accuracy": stacker_cv_accuracy,
        "stacker_cv_by_l2": {str(key): value
                              for key, value in stacker_cv_by_l2.items()},
        "selected_l2": best_l2,
        "temperatures": temperatures,
        "parameters": parameters,
        "pair_logits": pair_logits,
        "choices": choices,
    }
