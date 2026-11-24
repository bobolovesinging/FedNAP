"""Loss helpers shared by reward-choice trainers."""

import torch


def cal_choice_loss(logits, labels, choices, ignore_index):
    """Return ``None`` when truncation removed every choice target."""
    # Select the tiny choice vocabulary before materializing shifted logits.
    new_logits = logits[..., :-1, choices].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    new_labels = torch.full_like(shift_labels, int(ignore_index))
    for index, choice in enumerate(choices):
        new_labels[shift_labels == choice] = index
    if not torch.any(new_labels != int(ignore_index)).item():
        return new_logits, new_labels, None
    loss = torch.nn.functional.cross_entropy(
        new_logits.view(-1, len(choices)),
        new_labels.view(-1),
        ignore_index=int(ignore_index),
    )
    return new_logits, new_labels, loss
