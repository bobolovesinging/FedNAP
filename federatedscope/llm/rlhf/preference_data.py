import json
import math


def should_load_selector(preference_data_file):
    return not bool(preference_data_file)


def load_preference_data(preference_data_file):
    with open(preference_data_file, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    if not isinstance(data, list):
        raise ValueError("Preference data must be a JSON list")

    required_fields = {"output_A", "output_B", "choice"}
    weight_presence = ["dpo_weight" in sample for sample in data]
    if any(weight_presence) and not all(weight_presence):
        raise ValueError(
            "dpo_weight must be present on all preference samples or none")
    for index, sample in enumerate(data):
        missing = required_fields.difference(sample)
        if missing:
            raise ValueError(
                f"Preference sample {index} is missing fields: "
                f"{sorted(missing)}")
        if sample["choice"] not in (0, 1):
            raise ValueError(
                f"Preference sample {index} has invalid choice: "
                f"{sample['choice']!r}")
        if weight_presence and weight_presence[0]:
            weight = float(sample["dpo_weight"])
            if not math.isfinite(weight):
                raise ValueError(
                    f"Preference sample {index} weight must be finite")
            if weight <= 0.0:
                raise ValueError(
                    f"Preference sample {index} weight must be positive")
            if weight > 1.0:
                raise ValueError(
                    f"Preference sample {index} weight must be at most 1")

    return data


def verify_shp_pool(records, staged_dir):
    """Require every expected completion-index pair for each staged prompt."""
    from itertools import combinations
    from pathlib import Path
    from scripts.prepare_server_validation import file_sha256

    root = Path(staged_dir)
    sources = {row.get("source") for row in records}
    if len(sources) != 1 or not sources <= {"reddit-2", "reddit-4", "saferlhf-2"}:
        raise ValueError("Expected one staged SHP candidate source")
    pool, count = sources.pop().split("-")
    relative = f"prompts/{pool}.json"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest["files"][relative]["sha256"]
    if file_sha256(root / relative) != expected_hash:
        raise ValueError("SHP candidate file differs from staging manifest")
    prompts = json.loads((root / relative).read_text(encoding="utf-8"))
    expected = {(p["prompt_id"], p["instruction"], i, j) for p in prompts
                for i, j in combinations(range(int(count)), 2)}
    for row in records:
        key = (row["prompt_id"], row["instruction"],
               row["completion_index_A"], row["completion_index_B"])
        if row["source_sha256"] != expected_hash or key not in expected:
            raise ValueError("Unexpected or duplicate SHP candidate pair")
        expected.remove(key)
    if expected:
        raise ValueError("Incomplete SHP candidate coverage")
