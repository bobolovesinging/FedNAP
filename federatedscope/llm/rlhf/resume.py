import os

import torch


def load_policy_checkpoint(model,
                           checkpoint_path,
                           expected_start_round=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Policy checkpoint must be a dictionary.")
    if "cur_round" not in checkpoint or "model" not in checkpoint:
        raise ValueError(
            "Policy checkpoint must contain cur_round and model.")
    if not isinstance(checkpoint["cur_round"], int) or checkpoint[
            "cur_round"] < 0:
        raise ValueError("Policy checkpoint cur_round must be non-negative.")
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("Policy checkpoint model must be a state dictionary.")

    start_round = checkpoint["cur_round"] + 1
    if expected_start_round is not None and start_round != \
            expected_start_round:
        raise ValueError(
            f"Policy checkpoint starts at round {start_round}; "
            f"expected next round {expected_start_round}.")

    model.load_state_dict(checkpoint["model"])
    return start_round


def training_rounds(start_round, total_round_num):
    if start_round < 0 or start_round > total_round_num:
        raise ValueError(
            f"Invalid training round range: {start_round}..{total_round_num}")
    return range(start_round, total_round_num)
