"""Read nanochatn't settings from a TOML file alongside this module."""

import argparse
from pathlib import Path
import tomllib

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.toml")

REQUIRED = {
    "dataset": ("name", "train_shards", "download_workers"),
    "tokenizer": ("vocab_size", "max_chars", "doc_cap"),
    "packing": ("row_len", "workers"),
}


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser()
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    for section, keys in REQUIRED.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing [{section}] section in {path}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing {section}.{key} in {path}")

    if not isinstance(config["dataset"]["name"], str) or not config["dataset"]["name"].strip():
        raise ValueError("dataset.name must be a nonempty string")

    for section, keys in REQUIRED.items():
        for key in keys:
            if section == "dataset" and key == "name":
                continue
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")

    if not 261 <= config["tokenizer"]["vocab_size"] < 65535:
        raise ValueError("tokenizer.vocab_size must be between 261 and 65534")
    return config


def cli_config(parser):
    """Load --config before defining argparse defaults; CLI args can override them."""
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = early.parse_known_args()
    config = load_config(known.config)
    parser.add_argument("--config", type=Path, default=known.config,
                        help="TOML configuration file (default: consumer/config.toml)")
    parser.add_argument("--show-config", action="store_true",
                        help="Print effective settings and exit without changing data")
    return config


def print_effective_args(args):
    print("Effective settings (no work performed):")
    for name, value in vars(args).items():
        if name != "show_config":
            print(f"  {name} = {value}")
