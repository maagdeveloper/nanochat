"""Delegate tokenizer training and pretraining packing to native nanochat."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from consumer.artifacts import (
    NATIVE_PROVENANCE_LOCK,
    PACKED_LOCK,
    RAW_INPUTS_LOCK,
    TOKENIZER_LOCK,
    create_packing_view,
    plan_artifacts,
    require_locked_packed,
    require_locked_raw_inputs,
    require_locked_tokenizer,
    require_native_provenance,
    seal_native_provenance,
    _native_sources_snapshot,
    seal_verified_packed,
    seal_verified_tokenizer,
)
from consumer.config import cli_config, print_effective_args


# Intent record for an interrupted *consumer-initiated* native preparation.
# This is a recovery marker, not cryptographic proof of output provenance.
PENDING_PREPARATION = ".consumer_prepare_pending.json"


def pending_record(plan):
    raw_lock = Path(plan["tokenizer"]) / RAW_INPUTS_LOCK
    return {
        "schema": 3,
        "tokenizer_name": plan["tokenizer_name"],
        "packed_name": plan["packed_name"],
        "raw_lock_sha256": hashlib.sha256(raw_lock.read_bytes()).hexdigest(),
        "native_source_sha256": _native_sources_snapshot()["source_sha256"],
    }


def check_pending(path, expected):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Invalid preparation marker: {path}")
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid preparation marker: {path}") from exc
    if recorded != expected:
        raise RuntimeError(f"Preparation marker does not match this run: {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare pretraining shards with native nanochat")
    cfg = cli_config(parser)
    parser.add_argument("--dataset", default=cfg["dataset"]["name"])
    parser.add_argument("-n", "--train-shards", type=int,
                        default=cfg["dataset"]["train_shards"])
    parser.add_argument("--row-len", type=int, default=cfg["packing"]["row_len"])
    parser.add_argument("--workers", type=int, default=cfg["packing"]["workers"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the native command without creating artifacts")
    args = parser.parse_args()

    if args.show_config:
        print_effective_args(args)
        return
    if args.train_shards < 1 or args.row_len < 1 or args.workers < 1:
        parser.error("--train-shards, --row-len and --workers must be positive")

    effective = {
        **cfg,
        "dataset": {**cfg["dataset"], "name": args.dataset,
                    "train_shards": args.train_shards},
        "packing": {**cfg["packing"], "row_len": args.row_len,
                    "workers": args.workers},
    }
    plan = plan_artifacts(effective)

    command = [
        sys.executable, "-u", "-m", "scripts.base_prepare",
        "-n", str(args.train_shards),
        "-T", str(args.row_len),
        "-w", str(args.workers),
        "--vocab-size", str(effective["tokenizer"]["vocab_size"]),
        "--max-chars", str(effective["tokenizer"]["max_chars"]),
        "--doc-cap", str(effective["tokenizer"]["doc_cap"]),
    ]

    print(f"Packed view: {plan['packed']}")
    print(f"NANOCHAT_DATASET={plan['packed_name']}")
    print("Native command:", shlex.join(command))

    if args.dry_run:
        print("DRY RUN: no packing or view creation performed")
        return

    # Consumer creates views and verifies inputs; nanochat performs preparation.
    view = Path(create_packing_view(effective))
    require_locked_raw_inputs(effective)

    tokenizer_dir = Path(plan["tokenizer"]) / "tokenizer"
    tokenizer_present = (tokenizer_dir / "tokenizer.pkl").is_file()
    tokenizer_lock = Path(plan["tokenizer"]) / TOKENIZER_LOCK
    packed_lock = view / PACKED_LOCK
    native_lock = view / NATIVE_PROVENANCE_LOCK
    pending = Path(plan["tokenizer"]) / PENDING_PREPARATION
    expected = pending_record(plan)

    resuming = pending.exists() or pending.is_symlink()
    if resuming:
        check_pending(pending, expected)
        print("Resuming interrupted native preparation", flush=True)

    # Existing tokenizer files must be verified or belong to our interrupted run.
    if tokenizer_lock.exists() or tokenizer_lock.is_symlink():
        require_locked_tokenizer(effective)
    elif tokenizer_present and not resuming:
        raise RuntimeError(f"Unverified tokenizer without preparation marker: {tokenizer_dir}")

    # Never silently bless pre-existing .bin payloads without a packed lock.
    if packed_lock.exists() or packed_lock.is_symlink():
        require_locked_packed(effective)
    elif list(view.glob("*.bin")) and not resuming:
        raise RuntimeError(f"Unverified packed shards without preparation marker: {view}")

    # Never backfill provenance for pre-existing outputs without an active
    # consumer preparation marker. Existing provenance must match current code.
    if native_lock.exists() or native_lock.is_symlink():
        require_native_provenance(effective)
    elif packed_lock.is_file() and not resuming:
        raise RuntimeError(f"Existing packed artifacts have no native provenance: {view}")

    # Mark the entire native preparation as in progress, until all locks exist.
    if not resuming and not packed_lock.is_file():
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, sort_keys=True)
            handle.write("\n")

    env = os.environ.copy()
    env["NANOCHAT_DATASET"] = view.name
    root = Path(__file__).resolve().parents[2]
    subprocess.run(command, cwd=root, env=env, check=True)

    if not tokenizer_lock.is_file():
        seal_verified_tokenizer(effective)
    require_locked_tokenizer(effective)

    if not packed_lock.is_file():
        seal_verified_packed(effective)
    require_locked_packed(effective)

    # A successful native run is not enough to claim provenance if the source
    # changed during execution. Keep the marker so recovery remains explicit.
    if pending.exists() or pending.is_symlink():
        check_pending(pending, expected)
    if not native_lock.is_file():
        if not pending.is_file():
            raise RuntimeError(f"Cannot establish native provenance without preparation marker: {view}")
        seal_native_provenance(effective)
    require_native_provenance(effective)

    # Clear recovery intent only after all three output locks verify.
    pending.unlink(missing_ok=True)
    print("Verified native preparation: tokenizer, packed, and native provenance locks", flush=True)


if __name__ == "__main__":
    main()
