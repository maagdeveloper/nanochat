"""Train a rustbpe/tiktoken tokenizer with the original nanochat settings/algorithm."""

import argparse
import os
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from consumer.config import cli_config, print_effective_args
from harness.dataset import list_parquet_files
from nanochat.common import get_dataset_dir
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer_dir, tokenizer_id
from consumer.artifacts import create_tokenizer_view

def build_token_bytes(tokenizer):
    special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
    return torch.tensor([
        0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
        for i in range(tokenizer.get_vocab_size())
    ], dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser(description="Train nanochat's BPE tokenizer")
    cfg = cli_config(parser)
    parser.add_argument("--dataset", default=cfg["dataset"]["name"])
    parser.add_argument("-n", "--train-shards", type=int,
                        default=cfg["dataset"]["train_shards"])
    parser.add_argument("--vocab-size", type=int,
                        default=cfg["tokenizer"]["vocab_size"])
    parser.add_argument("--max-chars", type=int,
                        default=cfg["tokenizer"]["max_chars"])
    parser.add_argument("--doc-cap", type=int,
                        default=cfg["tokenizer"]["doc_cap"])
    args = parser.parse_args()

    if args.show_config:
        print_effective_args(args)
        return

    if not 261 <= args.vocab_size < 65535:
        parser.error("--vocab-size must be between 261 and 65534")
    if args.max_chars < 1 or args.doc_cap < 1 or args.train_shards < 1:
        parser.error("--max-chars, --doc-cap and --train-shards must be positive")

    # Resolve artifact identity from the effective CLI + TOML settings.
    effective = {
        **cfg,
        "dataset": {
            **cfg["dataset"],
            "name": args.dataset,
            "train_shards": args.train_shards,
        },
        "tokenizer": {
            **cfg["tokenizer"],
            "vocab_size": args.vocab_size,
            "max_chars": args.max_chars,
            "doc_cap": args.doc_cap,
        },
    }

    view_dir = create_tokenizer_view(effective)

    # nanochat resolves its tokenizer location via NANOCHAT_DATASET.
    os.environ["NANOCHAT_DATASET"] = view_dir.name

    parquets = list_parquet_files(str(view_dir))
    if len(parquets) < 2:
        raise RuntimeError("Dataset needs train and validation parquet shards")
    if len(parquets) - 1 < args.train_shards:
        raise RuntimeError(f"Found only {len(parquets) - 1} training shards")
    train_parquets = parquets[:-1][:args.train_shards]

    tok_dir = Path(get_tokenizer_dir())
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "tokenizer.pkl"
    byte_path = tok_dir / "token_bytes.pt"

    if tok_path.exists():
        print("Tokenizer exists — loading; training settings do not retrain it.")
        tokenizer = RustBPETokenizer.from_directory(str(tok_dir))
        if tokenizer.get_vocab_size() != args.vocab_size:
            raise ValueError("Existing tokenizer has a different vocab size. Use a separate dataset or clear its tokenizer artifacts deliberately.")
    else:
        if byte_path.exists():
            raise RuntimeError(f"{byte_path} exists without tokenizer.pkl")

        def text_iterator():
            """Same row-group/doc order and stopping condition as nanochat base_prepare.py."""
            nchars = 0
            for parquet_path in train_parquets:
                parquet = pq.ParquetFile(parquet_path)
                for group in range(parquet.num_row_groups):
                    texts = parquet.read_row_group(group).column("text").to_pylist()
                    for doc in texts:
                        text = doc[:args.doc_cap]
                        nchars += len(text)
                        yield text
                        if nchars > args.max_chars:
                            return

        print("Training tokenizer...")
        start = time.perf_counter()
        tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
        assert tokenizer.get_vocab_size() == args.vocab_size

        with tempfile.TemporaryDirectory(dir=tok_dir.parent) as temp:
            temporary = Path(temp)
            tokenizer.save(str(temporary))
            torch.save(build_token_bytes(tokenizer), temporary / "token_bytes.pt")
            os.replace(temporary / "tokenizer.pkl", tok_path)
            os.replace(temporary / "token_bytes.pt", byte_path)
        print(f"Training time: {time.perf_counter() - start:.2f}s")

    if not byte_path.exists():
        torch.save(build_token_bytes(tokenizer), byte_path)

    for sample in ("Hello world!", "بسم الله", "Once upon a time..."):
        assert tokenizer.decode(tokenizer.encode(sample)) == sample

    print("Vocabulary:", tokenizer.get_vocab_size())
    print("Tokenizer ID:", tokenizer_id(str(tok_dir)))
    print("Artifacts:", tok_dir)
    print("Roundtrip: PASSED")
    print("Tokenizer ready!")


if __name__ == "__main__":
    main()
