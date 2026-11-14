"""Prepare/verify raw pretraining parquet shards using nanochat's original downloader."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from consumer.config import cli_config, print_effective_args
from harness.dataset import (
    MAX_SHARD,
    index_to_filename,
    list_parquet_files,
    materialize,
)
from nanochat.common import CANONICAL_DATASET, get_dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare nanochat pretraining data")
    cfg = cli_config(parser)
    parser.add_argument("--dataset", default=cfg["dataset"]["name"])
    parser.add_argument("-n", "--train-shards", type=int,
                        default=cfg["dataset"]["train_shards"])
    parser.add_argument("--workers", type=int,
                        default=cfg["dataset"]["download_workers"])
    args = parser.parse_args()

    if args.show_config:
        print_effective_args(args)
        return

    if args.train_shards < 1 or args.workers < 1:
        parser.error("--train-shards and --workers must be positive")
    if args.dataset == CANONICAL_DATASET and args.train_shards > MAX_SHARD:
        parser.error(f"--train-shards must be <= {MAX_SHARD}")

    materialize(args.dataset, args.train_shards, num_workers=args.workers)
    dataset_dir = Path(get_dataset_dir(args.dataset))

    if args.dataset == CANONICAL_DATASET:
        paths = [dataset_dir / index_to_filename(i)
                 for i in range(args.train_shards)]
        paths.append(dataset_dir / index_to_filename(MAX_SHARD))
    else:
        parquet_paths = list_parquet_files(str(dataset_dir))
        if len(parquet_paths) - 1 < args.train_shards:
            raise ValueError(f"Only {len(parquet_paths) - 1} training shards found")
        paths = [Path(path) for path in parquet_paths[:args.train_shards]]
        paths.append(Path(parquet_paths[-1]))

    total_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        if "text" not in parquet.schema.names:
            raise ValueError(f"Missing 'text' column: {path}")
        count = parquet.metadata.num_rows
        total_rows += count
        print(f"{path.name}: {count:,} documents")

    print(f"\nDataset: {args.dataset}")
    print(f"Files: {len(paths)}")
    print(f"Documents: {total_rows:,}")
    print(f"Path: {dataset_dir}")


if __name__ == "__main__":
    main()
