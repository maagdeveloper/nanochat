
import hashlib
import json
import os
import tempfile
from pathlib import Path

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


def _selected_raw_sources(cfg):
    """Select the native train shards and final validation shard."""
    from harness.dataset import MAX_SHARD, index_to_filename, list_parquet_files
    from nanochat.common import CANONICAL_DATASET

    plan = plan_artifacts(cfg)
    raw_dir = Path(plan["raw"])
    num_train = cfg["dataset"]["train_shards"]
    if num_train < 1:
        raise ValueError("train_shards must be positive")

    if cfg["dataset"]["name"] == CANONICAL_DATASET:
        if num_train > MAX_SHARD:
            raise ValueError("Too many ClimbMix training shards")
        sources = [raw_dir / index_to_filename(i) for i in range(num_train)]
        sources.append(raw_dir / index_to_filename(MAX_SHARD))
    else:
        files = [Path(p) for p in list_parquet_files(str(raw_dir))]
        if len(files) < num_train + 1:
            raise ValueError("Not enough train/validation shards")
        sources = files[:num_train] + files[-1:]
    return sources


def create_tokenizer_view(cfg):
    """Link selected raw shards into a versioned tokenizer dataset."""
    import os
    from pathlib import Path

    plan = plan_artifacts(cfg)
    view_dir = Path(plan["tokenizer"])
    sources = _selected_raw_sources(cfg)

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

    # A config fingerprint is not enough: bind existing views to the actual
    # input bytes so native base_prepare cannot silently reuse stale outputs.
    verify_raw_inputs(cfg, view_dir, sources)

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

    # Native base_prepare trains a missing tokenizer itself. Reject partial
    # artifacts, since it skips training when tokenizer.pkl already exists.
    if tokenizer_dir.is_symlink() or (tokenizer_dir.exists() and not tokenizer_dir.is_dir()):
        raise RuntimeError(f"Invalid tokenizer directory: {tokenizer_dir}")
    has_tokenizer = (tokenizer_dir / "tokenizer.pkl").is_file()
    has_token_bytes = (tokenizer_dir / "token_bytes.pt").is_file()
    if has_tokenizer != has_token_bytes:
        raise RuntimeError(f"Incomplete tokenizer artifacts: {tokenizer_dir}")

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
            # The directory may be absent on a fresh dataset.
            valid_source = source.is_dir() or not source.exists()
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

    # Make the symlink target before linking it into the packing view.
    # The native script will populate this empty directory if necessary.
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
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

RAW_INPUTS_LOCK = ".consumer_raw_inputs.json"


def _snapshot_sources(cfg, sources):
    """Content hashes of exactly the raw shards used by this tokenizer view."""
    entries = []
    for source in sources:
        h = hashlib.sha256()
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        entries.append({
            "name": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": h.hexdigest(),
        })
    return {
        "schema": 1,
        "dataset": cfg["dataset"]["name"],
        "train_shards": cfg["dataset"]["train_shards"],
        "sources": entries,
    }


def _write_lock(path, snapshot):
    """Write the new lock atomically, after source validation."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".raw-lock-", delete=False) as f:
        temporary = Path(f.name)
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_raw_inputs(cfg, view_dir, sources):
    """Fail closed if a versioned view's recorded input bytes have changed."""
    lock = view_dir / RAW_INPUTS_LOCK
    current = _snapshot_sources(cfg, sources)
    if lock.is_file():
        if json.loads(lock.read_text()) != current:
            raise RuntimeError(
                f"Raw inputs changed for {view_dir.name}; refusing stale artifacts. "
                "Do not overwrite this lock; use a new dataset identity."
            )
        return
    if lock.exists():
        raise RuntimeError(f"Invalid raw input lock: {lock}")

    # Never silently bless outputs created before we started recording hashes.
    tokenizer_dir = view_dir / "tokenizer"
    packed_dir = Path(plan_artifacts(cfg)["packed"])
    has_old_tokenizer = tokenizer_dir.is_dir() and any(tokenizer_dir.iterdir())
    has_old_packing = packed_dir.is_dir() and any(packed_dir.glob("*.bin"))
    if has_old_tokenizer or has_old_packing:
        raise RuntimeError(
            f"Existing artifacts have no raw-input lock: {view_dir}. "
            "Explicitly adopt them only after verifying the existing outputs."
        )

    view_dir.mkdir(parents=True, exist_ok=True)
    _write_lock(lock, current)


def require_locked_raw_inputs(cfg):
    """Read-only integrity check for consumers of an existing prepared view.

    Unlike create_tokenizer_view(), this never creates locks, directories, or
    symlinks. Training must reject unverified or missing preparation artifacts.
    """
    plan = plan_artifacts(cfg)
    view = Path(plan["tokenizer"])
    lock = view / RAW_INPUTS_LOCK
    if not lock.is_file():
        raise RuntimeError(f"Missing raw-input lock: {lock}")

    sources = _selected_raw_sources(cfg)
    expected = {source.name for source in sources}
    if {path.name for path in view.glob("*.parquet")} != expected:
        raise RuntimeError(f"Unexpected raw shard selection in {view}")

    for source in sources:
        link = view / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise RuntimeError(f"Incorrect raw shard link: {link}")

    # The lock is known to exist, so verify_raw_inputs only reads and compares.
    verify_raw_inputs(cfg, view, sources)
    return lock


def raw_dataset_snapshot(cfg):
    """Compute current content hashes for the selected tokenizer-view Parquets."""
    from harness.dataset import list_parquet_files

    view = Path(plan_artifacts(cfg)["tokenizer"])
    files = [Path(path) for path in list_parquet_files(str(view))]
    if len(files) != cfg["dataset"]["train_shards"] + 1:
        raise RuntimeError("Unexpected number of parquet shards")
    return _snapshot_sources(cfg, files)


def adopt_verified_raw_inputs(cfg):
    """Explicit one-time adoption of an existing, independently verified view."""
    view = Path(plan_artifacts(cfg)["tokenizer"])
    if not view.is_dir():
        raise FileNotFoundError(view)
    lock = view / RAW_INPUTS_LOCK
    if lock.exists():
        raise FileExistsError(lock)
    snapshot = raw_dataset_snapshot(cfg)
    _write_lock(lock, snapshot)
    return lock


# Tokenizer outputs are not identified by their config-derived view name alone.
TOKENIZER_LOCK = ".consumer_tokenizer.json"
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_tokenizer(cfg):
    """Bind both tokenizer outputs, native tokenizer ID, and the raw-input lock."""
    from nanochat.tokenizer import tokenizer_id

    view = Path(plan_artifacts(cfg)["tokenizer"])
    tokenizer_dir = view / "tokenizer"
    raw_lock = view / RAW_INPUTS_LOCK
    if not raw_lock.is_file():
        raise RuntimeError(f"Missing raw-input lock: {raw_lock}")

    files = {}
    for filename in TOKENIZER_FILES:
        path = tokenizer_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing tokenizer artifact: {path}")
        files[filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    return {
        "schema": 1,
        "raw_inputs_lock_sha256": _sha256_file(raw_lock),
        "tokenizer_id": tokenizer_id(str(tokenizer_dir)),
        "files": files,
    }


def seal_verified_tokenizer(cfg):
    """Explicitly seal a fresh or independently verified tokenizer; never overwrite."""
    require_locked_raw_inputs(cfg)
    view = Path(plan_artifacts(cfg)["tokenizer"])
    lock = view / TOKENIZER_LOCK
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(lock)
    _write_lock(lock, _snapshot_tokenizer(cfg))
    return lock


def require_locked_tokenizer(cfg):
    """Read-only check; missing or changed tokenizer files fail closed."""
    require_locked_raw_inputs(cfg)
    view = Path(plan_artifacts(cfg)["tokenizer"])
    lock = view / TOKENIZER_LOCK
    if not lock.is_file() or lock.is_symlink():
        raise RuntimeError(f"Missing or invalid tokenizer lock: {lock}")
    if json.loads(lock.read_text()) != _snapshot_tokenizer(cfg):
        raise RuntimeError(
            f"Tokenizer artifacts changed for {view.name}; refusing stale artifacts."
        )
    return lock


# Packed outputs are checked independently of their native headers. The native
# loader uses row length, vocabulary, and tokenizer ID for compatibility; those
# headers cannot detect changes elsewhere in a shard's payload.
PACKED_LOCK = ".consumer_packed.json"


def _snapshot_packed(cfg):
    """Snapshot exactly the packed shards that native base_train will consume."""
    from nanochat.dataloader import read_header
    from nanochat.tokenizer import tokenizer_id

    plan = plan_artifacts(cfg)
    view = Path(plan["packed"])
    tok_dir = Path(plan["tokenizer"]) / "tokenizer"
    tok_lock = Path(plan["tokenizer"]) / TOKENIZER_LOCK

    # A packed view must still point to the selected raw sources and tokenizer.
    tok_link = view / "tokenizer"
    if not tok_link.is_symlink() or tok_link.resolve() != tok_dir.resolve():
        raise RuntimeError(f"Incorrect packed tokenizer link: {tok_link}")
    for source in _selected_raw_sources(cfg):
        link = view / source.name
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise RuntimeError(f"Incorrect packed raw shard link: {link}")

    expected = [
        f"train_{i:05d}.bin"
        for i in range(cfg["dataset"]["train_shards"])
    ] + ["val_00000.bin"]
    found = {p.name for p in view.glob("*.bin")}
    if found != set(expected):
        raise RuntimeError(
            f"Incomplete or unexpected packed shards in {view}: "
            f"expected {sorted(expected)}, found {sorted(found)}"
        )

    native_tok_id = tokenizer_id(str(tok_dir))
    files = {}
    for name in expected:
        path = view / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or invalid packed artifact: {path}")
        row_len, rows, vocab, tok_id = read_header(str(path))
        if (row_len != cfg["packing"]["row_len"]
                or vocab != cfg["tokenizer"]["vocab_size"]
                or tok_id != native_tok_id or rows < 1):
            raise RuntimeError(f"Incompatible packed shard header: {path}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "header": [row_len, rows, vocab, tok_id],
        }

    return {
        "schema": 1,
        "packed_name": plan["packed_name"],
        "tokenizer_lock_sha256": _sha256_file(tok_lock),
        "row_len": cfg["packing"]["row_len"],
        "files": files,
    }


def seal_verified_packed(cfg):
    """Explicitly seal native packed results after a verified successful run.

    For legacy outputs, the caller must independently verify their origin
    before invoking this function. This function is not a parity test.
    """
    require_locked_tokenizer(cfg)
    view = Path(plan_artifacts(cfg)["packed"])
    lock = view / PACKED_LOCK
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(lock)
    _write_lock(lock, _snapshot_packed(cfg))
    return lock


def require_locked_packed(cfg):
    """Read-only verification before training or reusing packed outputs."""
    require_locked_tokenizer(cfg)
    view = Path(plan_artifacts(cfg)["packed"])
    lock = view / PACKED_LOCK
    if not lock.is_file() or lock.is_symlink():
        raise RuntimeError(f"Missing or invalid packed lock: {lock}")
    try:
        recorded = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid packed lock: {lock}") from exc
    if recorded != _snapshot_packed(cfg):
        raise RuntimeError(
            f"Packed artifacts changed for {view.name}; refusing stale artifacts."
        )
    return lock



# Native implementation provenance is deliberately separate from output integrity.
# A Git commit identifies historical context; source hashes detect relevant code
# changes without rejecting a cache because of an unrelated commit.
NATIVE_PROVENANCE_LOCK = ".consumer_native.json"
NATIVE_PREPARATION_SOURCES = (
    "scripts/base_prepare.py",
    "nanochat/tokenizer.py",
    "nanochat/dataloader.py",
    "harness/dataset.py",
    "nanochat/common.py",
)


def _native_root():
    return Path(__file__).resolve().parents[1]


def _native_head(root):
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _native_sources_snapshot():
    """Record actual code bytes, including edits not captured by Git HEAD."""
    root = _native_root()
    files = {}
    for name in NATIVE_PREPARATION_SOURCES:
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing native preparation source: {source}")
        files[name] = _sha256_file(source)
    return {"git_head": _native_head(root), "source_sha256": files}


def _native_provenance_snapshot(cfg):
    """Bind the implementation record to the already verified packed lock."""
    packed_lock = require_locked_packed(cfg)
    return {
        "schema": 1,
        "packed_lock_sha256": _sha256_file(packed_lock),
        "native": _native_sources_snapshot(),
    }


def seal_native_provenance(cfg):
    """Explicitly record native provenance after trusted preparation/parity review.

    Existing caches must NOT be sealed automatically just because their packed
    outputs passed integrity checks: that would misattribute their origin.
    """
    lock = Path(plan_artifacts(cfg)["packed"]) / NATIVE_PROVENANCE_LOCK
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(lock)
    _write_lock(lock, _native_provenance_snapshot(cfg))
    return lock


def require_native_provenance(cfg):
    """Read-only check: refuse a changed native implementation or packed lock.

    Git HEAD is recorded for historical context, but it is intentionally NOT
    compared: unrelated commits should not invalidate otherwise identical code.
    """
    lock = Path(plan_artifacts(cfg)["packed"]) / NATIVE_PROVENANCE_LOCK
    if not lock.is_file() or lock.is_symlink():
        raise RuntimeError(f"Missing or invalid native provenance: {lock}")
    try:
        recorded = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid native provenance: {lock}") from exc
    current = _native_provenance_snapshot(cfg)
    if (
        recorded.get("schema") != current["schema"]
        or recorded.get("packed_lock_sha256") != current["packed_lock_sha256"]
        or recorded.get("native", {}).get("source_sha256")
        != current["native"]["source_sha256"]
    ):
        raise RuntimeError(
            "Native preparation implementation or packed lock changed; "
            "cannot silently reuse these artifacts as outputs of the current implementation."
        )
    return lock

if __name__ == "__main__":
    plan = plan_artifacts(load_config())

    for key, value in plan.items():
        print(f"{key}: {value}")
