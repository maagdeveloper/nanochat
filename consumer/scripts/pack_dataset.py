"""Pack existing parquet shards using nanochat's original BOS-aligned pack_rows."""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from consumer.config import cli_config, print_effective_args
from harness.dataset import list_parquet_files
from nanochat.dataloader import HEADER_BYTES, pack_rows, read_header, write_shard
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer_dir, tokenizer_id
from consumer.artifacts import create_packing_view

TOKENIZER_THREADS = 4  # Matches nanochat base_prepare.py


def init_worker(tokenizer_dir):
    global worker_tokenizer
    worker_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)


def pack_shard(job):
    parquet_path, output_path, row_len, vocab_size, tok_id = job
    capacity = row_len + 1
    bos = worker_tokenizer.get_bos_token_id()
    raw_tokens = 0

    def documents():
        nonlocal raw_tokens
        parquet = pq.ParquetFile(parquet_path)
        for rg_idx in range(parquet.num_row_groups):
            texts = parquet.read_row_group(rg_idx).column("text").to_pylist()
            batches = worker_tokenizer.encode(
                texts, prepend=bos, num_threads=TOKENIZER_THREADS
            )
            for ids in batches:
                raw_tokens += len(ids)
                yield ids[:capacity]

    rows = np.stack([
        np.asarray(row, dtype=np.uint16)
        for row in pack_rows(documents(), capacity)
    ])
    write_shard(str(output_path), rows[:, :-1], rows[:, 1:], vocab_size, tok_id)
    return output_path.name, len(rows), raw_tokens


def main():
    parser = argparse.ArgumentParser(description="Pack nanochat pretraining shards")
    cfg = cli_config(parser)
    parser.add_argument("--dataset", default=cfg["dataset"]["name"])
    parser.add_argument("-n", "--train-shards", type=int,
                        default=cfg["dataset"]["train_shards"])
    parser.add_argument("--row-len", type=int,
                        default=cfg["packing"]["row_len"])
    parser.add_argument("--workers", type=int,
                        default=cfg["packing"]["workers"])
    args = parser.parse_args()

    if args.show_config:
        print_effective_args(args)
        return

    if args.row_len < 1 or args.workers < 1 or args.train_shards < 1:
        parser.error("--row-len, --workers and --train-shards must be positive")

    # Resolve artifact paths from effective CLI + TOML settings.
    effective = {
        **cfg,
        "dataset": {
            **cfg["dataset"],
            "name": args.dataset,
            "train_shards": args.train_shards,
        },
        "packing": {
            **cfg["packing"],
            "row_len": args.row_len,
            "workers": args.workers,
        },
    }

    dataset_dir = Path(create_packing_view(effective))

    # Let nanochat resolve the tokenizer from the active packing view.
    os.environ["NANOCHAT_DATASET"] = dataset_dir.name
    tok_dir = Path(get_tokenizer_dir())
    tokenizer = RustBPETokenizer.from_directory(str(tok_dir))
    vocab_size = tokenizer.get_vocab_size()
    tok_id = tokenizer_id(str(tok_dir))

    parquets = list_parquet_files(str(dataset_dir))
    if len(parquets) < 2:
        raise RuntimeError("Dataset needs train and validation shards")
    if len(parquets) - 1 < args.train_shards:
        raise RuntimeError(f"Only {len(parquets) - 1} training shards found")

    train_parquets = parquets[:-1][:args.train_shards]
    val_parquet = parquets[-1]
    jobs = [
        (path, dataset_dir / f"train_{i:05d}.bin", args.row_len, vocab_size, tok_id)
        for i, path in enumerate(train_parquets)
    ]
    jobs.append((val_parquet, dataset_dir / "val_00000.bin",
                 args.row_len, vocab_size, tok_id))

    pending = []
    for job in jobs:
        path = job[1]
        if not path.exists():
            pending.append(job)
            continue
        existing_len, rows, existing_vocab, existing_id = read_header(str(path))
        if (existing_len, existing_vocab, existing_id) != (
            args.row_len, vocab_size, tok_id
        ):
            raise RuntimeError(f"Incompatible packed shard: {path}")
        expected_bytes = HEADER_BYTES + rows * args.row_len * 2 * np.dtype(np.uint16).itemsize
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Invalid shard size: {path}")
        print(f"SKIP: {path.name} ({rows:,} rows)")

    print(f"Tokenizer ID: {tok_id}")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Row length: {args.row_len:,}")
    print(f"Packing: {len(pending)}/{len(jobs)} shards")

    if pending:
        with Pool(args.workers, initializer=init_worker, initargs=(str(tok_dir),)) as pool:
            for name, rows, raw_tokens in pool.imap_unordered(pack_shard, pending):
                packed = rows * (args.row_len + 1)
                cropped = 100 * (1 - packed / raw_tokens)
                print(f"{name}: {rows:,} rows | {packed:,} packed tokens | {cropped:.1f}% cropped")

    print("Packing completed.")


if __name__ == "__main__":
    main()
