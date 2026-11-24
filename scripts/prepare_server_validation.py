"""Shared file and normalized-prompt identities for canonical task staging."""

import hashlib
import json
import unicodedata


def read_records(path):
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".json":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {path}")
            yield from records
        else:
            yield from (json.loads(line) for line in stream if line.strip())


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_id(record, prompt_field):
    # Released TLDR attribute annotations wrap the comparison under "data".
    fields = record.get("data", record) if prompt_field == "post" else record
    prompt = normalized(fields[prompt_field])
    if not prompt:
        raise ValueError("Expected nonempty prompt text")
    return prompt


def prompt_ids(path, prompt_field):
    ids = {prompt_id(record, prompt_field) for record in read_records(path)}
    if not ids:
        raise ValueError(f"Empty input: {path}")
    return ids


def comparison_rank(record, prompt_field):
    # Use the same SHA-256 ordering as the existing CPF calibration loader.
    fields = ["category", prompt_field]
    if prompt_field == "post":
        fields.append("title")
    fields.extend(["output_A", "output_B", "choice"])
    identity = json.dumps([record.get(field) for field in fields],
                          ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).digest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    raise SystemExit('Use scripts/stage_tldr.py or python -m scripts.stage_shp to build all roles together.')
