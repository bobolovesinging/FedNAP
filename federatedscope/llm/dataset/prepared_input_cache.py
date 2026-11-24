"""Opt-in, immutable cache for trusted local prepared CPU datasets."""
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _load(directory, fingerprint):
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = directory / "dataset.pickle"
    if manifest.get("complete") is not True or manifest.get("fingerprint") != fingerprint:
        raise ValueError("incompatible or incomplete cache manifest")
    if digest(payload) != manifest.get("payload_sha256"):
        raise ValueError("prepared input cache payload checksum mismatch")
    # Only read private, locally built experiment caches; never Hub/downloaded pickle.
    with payload.open("rb") as stream:
        return pickle.load(stream)


def load_or_build(root, fingerprint, builder):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        st = root.stat()
        if st.st_uid != os.getuid() or st.st_mode & 0o022:
            raise ValueError("cache root must be private to the current user")
    key = cache_key(fingerprint)
    directory = root / key
    started = time.monotonic()
    if directory.exists():
        value = _load(directory, fingerprint)
        logger.info("PREPARED_INPUT_CACHE_HIT key=%s seconds=%.3f",
                    key, time.monotonic() - started)
        return value
    lock = root / (key + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("cache build in progress; prepare cache before GPU submission") from error
    staging = root / (".pending-" + key + "-" + uuid.uuid4().hex)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        if directory.exists():
            return _load(directory, fingerprint)
        value = builder()
        staging.mkdir(mode=0o700)
        payload = staging / "dataset.pickle"
        with payload.open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(payload, 0o600)
        manifest = dict(complete=True, fingerprint=fingerprint,
                        payload_sha256=digest(payload),
                        build_seconds=time.monotonic() - started)
        with (staging / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, directory)
        logger.info("PREPARED_INPUT_CACHE_BUILD key=%s seconds=%.3f",
                    key, time.monotonic() - started)
        return value
    finally:
        lock.unlink()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def input_fingerprint(data_root, tokenizer, config, choice_format):
    import torch
    import transformers
    import tokenizers
    root = Path(data_root)
    llm = Path(__file__).parents[1]
    files = ["shp_train.jsonl", "shp_validation.jsonl", "shp_test.jsonl"]
    files += [name for name in ("manifest.json", "client_assignments.json")
              if (root / name).is_file()]
    sources = [llm / "dataloader/shp.py", llm / "dataset/llm_dataset.py",
               llm / "dataset/shp_preference.py", llm / "dataset/tldr_preference.py",
               llm / "dataloader/reddit_tldr.py", Path(__file__)]
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_state = dict(
        kind=type(tokenizer).__module__ + "." + type(tokenizer).__name__,
        vocabulary=tokenizer.get_vocab(), init=_plain(tokenizer.init_kwargs),
        special=_plain(tokenizer.special_tokens_map),
        model_max_length=int(tokenizer.model_max_length),
        padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side,
        pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id,
        backend=backend.to_str() if backend is not None else None)
    return dict(schema=1, files={name: digest(root / name) for name in files},
                source={str(p.relative_to(llm)): digest(p) for p in sources},
                tokenizer_sha256=cache_key(tokenizer_state),
                versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                              tokenizers=tokenizers.__version__),
                choice_format=bool(choice_format),
                clients=int(config.federate.client_num),
                pool=bool(getattr(config.data, "pool_prepared_train", False)),
                max_num_test=int(getattr(config.data, "max_num_test", -1)),
                complete_inputs=bool(getattr(config.data, "require_complete_pair_inputs", False)))
