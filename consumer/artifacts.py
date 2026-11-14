
import hashlib
import json

from consumer.config import load_config
from nanochat.common import get_dataset_dir


def fingerprint(settings):
    """Stable short ID for a reproducible configuration."""
    payload = json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:12]


def plan_artifacts(cfg):
    dataset = cfg["dataset"]
    tokenizer = cfg["tokenizer"]
    packing = cfg["packing"]

    source = dataset["name"]

    tokenizer_spec = {
        "recipe": "rustbpe-v1",
        "dataset": source,
        "train_shards": dataset["train_shards"],
        "vocab_size": tokenizer["vocab_size"],
        "max_chars": tokenizer["max_chars"],
        "doc_cap": tokenizer["doc_cap"],
    }

    tokenizer_name = (
        f"{source}__tok_{fingerprint(tokenizer_spec)}"
    )

    packing_spec = {
        "recipe": "bos-best-fit-crop-v1",
        "tokenizer": tokenizer_name,
        "row_len": packing["row_len"],
    }

    packed_name = (
        f"{source}__pack_{fingerprint(packing_spec)}"
    )

    return {
        "raw": get_dataset_dir(source),
        "tokenizer": get_dataset_dir(tokenizer_name),
        "packed": get_dataset_dir(packed_name),
        "tokenizer_name": tokenizer_name,
        "packed_name": packed_name,
    }


def create_tokenizer_view(cfg):
    """Link selected raw shards into a versioned tokenizer dataset."""
    import os
    from pathlib import Path

    from harness.dataset import (
        MAX_SHARD,
        index_to_filename,
        list_parquet_files,
    )
    from nanochat.common import CANONICAL_DATASET

    plan = plan_artifacts(cfg)

    raw_dir = Path(plan["raw"])
    view_dir = Path(plan["tokenizer"])
    num_train = cfg["dataset"]["train_shards"]

    if num_train < 1:
        raise ValueError("train_shards must be positive")

    if cfg["dataset"]["name"] == CANONICAL_DATASET:
        if num_train > MAX_SHARD:
            raise ValueError("Too many ClimbMix training shards")

        sources = [
            raw_dir / index_to_filename(i)
            for i in range(num_train)
        ]
        sources.append(raw_dir / index_to_filename(MAX_SHARD))

    else:
        files = [
            Path(p) for p in list_parquet_files(str(raw_dir))
        ]

        if len(files) < num_train + 1:
            raise ValueError("Not enough train/validation shards")

        sources = files[:num_train] + files[-1:]

    # Validate everything before creating links.
    expected = {src.name for src in sources}

    for src in sources:
        if not src.is_file():
            raise FileNotFoundError(src)

    if view_dir.is_symlink():
        raise RuntimeError(f"View directory is a symlink: {view_dir}")

    if view_dir.exists():
        existing = {p.name for p in view_dir.glob("*.parquet")}
        if existing - expected:
            raise RuntimeError("Unexpected Parquet files in view")

    for src in sources:
        link = view_dir / src.name

        if link.is_symlink():
            if link.resolve() != src.resolve():
                raise RuntimeError(f"Wrong symlink: {link}")
        elif link.exists():
            raise FileExistsError(link)

    view_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        link = view_dir / src.name

        if not link.is_symlink():
            relative = os.path.relpath(src, view_dir)
            link.symlink_to(relative)

    return view_dir


def create_packing_view(cfg):
    """Create a reusable dataset view for a specific packing configuration."""
    import os
    from pathlib import Path

    from harness.dataset import list_parquet_files

    plan = plan_artifacts(cfg)

    # Reuse the selected raw shards and versioned tokenizer.
    tokenizer_view = Path(create_tokenizer_view(cfg))
    tokenizer_dir = tokenizer_view / "tokenizer"
    packing_view = Path(plan["packed"])

    # A packing view must never use an incomplete tokenizer.
    for filename in ("tokenizer.pkl", "token_bytes.pt"):
        if not (tokenizer_dir / filename).is_file():
            raise FileNotFoundError(tokenizer_dir / filename)

    parquet_files = [
        Path(path)
        for path in list_parquet_files(str(tokenizer_view))
    ]

    expected_count = cfg["dataset"]["train_shards"] + 1
    if len(parquet_files) != expected_count:
        raise RuntimeError("Unexpected number of parquet shards")

    # Shared data and tokenizer; only packed binaries are new.
    targets = {
        path.name: path for path in parquet_files
    }
    targets["tokenizer"] = tokenizer_dir

    if packing_view.is_symlink():
        raise RuntimeError(f"Packing view is a symlink: {packing_view}")

    if packing_view.exists() and not packing_view.is_dir():
        raise FileExistsError(packing_view)

    if packing_view.exists():
        expected_parquets = {path.name for path in parquet_files}
        existing_parquets = {
            path.name for path in packing_view.glob("*.parquet")
        }
        if existing_parquets - expected_parquets:
            raise RuntimeError("Unexpected parquet files in packing view")

    # Validate existing paths before creating anything.
    for name, source in targets.items():
        if name == "tokenizer":
            valid_source = source.is_dir()
        else:
            valid_source = source.is_file()

        if not valid_source:
            raise FileNotFoundError(source)

        link = packing_view / name

        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"Incorrect symlink: {link}")
        elif link.exists():
            raise FileExistsError(link)

    packing_view.mkdir(parents=True, exist_ok=True)

    for name, source in targets.items():
        link = packing_view / name

        if not link.is_symlink():
            relative = os.path.relpath(source, packing_view)
            link.symlink_to(
                relative,
                target_is_directory=(name == "tokenizer"),
            )

    return packing_view

def raw_dataset_snapshot(cfg):
    """Compute content hashes for selected raw parquet shards."""
    from pathlib import Path
    from harness.dataset import list_parquet_files

    plan = plan_artifacts(cfg)
    view = Path(plan["tokenizer"])

    files = [
        Path(path)
        for path in list_parquet_files(str(view))
    ]

    expected = cfg["dataset"]["train_shards"] + 1
    if len(files) != expected:
        raise RuntimeError(
            f"Expected {expected} parquet files, found {len(files)}"
        )

    sources = []

    for path in files:
        hasher = hashlib.sha256()

        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)

        sources.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hasher.hexdigest(),
        })

    return {
        "dataset": cfg["dataset"]["name"],
        "train_shards": cfg["dataset"]["train_shards"],
        "sources": sources,
    }

if __name__ == "__main__":
    plan = plan_artifacts(load_config())

    for key, value in plan.items():
        print(f"{key}: {value}")
