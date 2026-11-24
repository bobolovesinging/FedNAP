import json
from pathlib import Path


def read_generation_json(path, test_file):
    """Bind every generated summary to the same official row in source order."""
    from scripts.prepare_server_validation import file_sha256
    from scripts.tldr_benchmark import TEST_SHA
    if file_sha256(Path(test_file)) != TEST_SHA:
        raise ValueError("Unexpected TLDR test file")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    summaries_only = isinstance(rows, dict)
    if summaries_only:
        if rows.get("test_sha256") != TEST_SHA:
            raise ValueError("Summaries do not identify the official TLDR test file")
        rows = rows.get("summaries")
        if not isinstance(rows, list) or any(not isinstance(summary, str) for summary in rows):
            raise ValueError("Expected a list of generated summaries")
    elif not isinstance(rows, list):
        raise ValueError("Expected a generation list or a summary release")
    with Path(test_file).open(encoding="utf-8") as stream:
        official = [json.loads(line) for line in stream]
    if len(official) != len(rows):
        raise ValueError("Incomplete TLDR evaluation coverage")
    result = []
    for index, (row, ref) in enumerate(zip(rows, official)):
        if summaries_only:
            completion = row
        else:
            if any(row[key] != ref[key] for key in ("subreddit", "title", "post", "summary")):
                raise ValueError(f"Generation does not match official test row {index}")
            completion = row["completion"]
        result.append(dict(title=ref["title"], subreddit=ref["subreddit"], post=ref["post"],
                           query="Summarize the following post\n\nTitle: {title}\n\nPost: {post}".format_map(ref),
                           response=completion, response_another=ref["summary"]))
    return result


def build_model_second_input(sample, input_builder):
    required = ("query", "response", "response_another")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(
            "Auto-J pairwise sample is missing required fields: "
            + ", ".join(missing))

    return input_builder(
        prompt=sample["query"],
        resp1=sample["response_another"],
        resp2=sample["response"],
        protocol="pairwise_tie",
    )


def summarize_model_second_choices(choices):
    choices = list(choices)
    valid = [choice for choice in choices if choice in (0, 1, 2)]
    model_wins = sum(choice == 1 for choice in valid)
    model_losses = sum(choice == 0 for choice in valid)
    ties = sum(choice == 2 for choice in valid)

    return {
        "records": len(choices),
        "valid": len(valid),
        "invalid": len(choices) - len(valid),
        "model_wins": model_wins,
        "model_losses": model_losses,
        "ties": ties,
        "model_win_rate": model_wins / len(valid) if valid else None,
    }
