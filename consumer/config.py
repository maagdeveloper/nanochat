"""Read and validate nanochatn't configuration from a TOML file."""

import argparse
import math
from pathlib import Path
import tomllib

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.toml")

REQUIRED = {
    "dataset": ("name", "train_shards", "download_workers"),
    "tokenizer": ("vocab_size", "max_chars", "doc_cap"),
    "packing": ("row_len", "workers"),
    "model": ("depth", "aspect_ratio", "head_dim", "max_seq_len", "window_pattern"),
    "training": (
        "model_tag", "device_type", "device_batch_size",
        "target_param_data_ratio", "warmup_steps",
        "eval_every", "eval_tokens", "save_every",
    ),
}


def _integer(section, key, value, *, minimum=1, allow_auto=False):
    if type(value) is not int:
        raise ValueError(f"{section}.{key} must be an integer")
    if allow_auto and value == -1:
        return
    if value < minimum:
        constraint = f">= {minimum}"
        if allow_auto:
            constraint += " or -1 (native automatic/disabled setting)"
        raise ValueError(f"{section}.{key} must be {constraint}")


def _nonempty_string(section, key, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a nonempty string")


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

    dataset = config["dataset"]
    tokenizer = config["tokenizer"]
    packing = config["packing"]
    model = config["model"]
    training = config["training"]

    _nonempty_string("dataset", "name", dataset["name"])
    for key in ("train_shards", "download_workers"):
        _integer("dataset", key, dataset[key])
    for key in ("vocab_size", "max_chars", "doc_cap"):
        _integer("tokenizer", key, tokenizer[key])
    if not 261 <= tokenizer["vocab_size"] < 65535:
        raise ValueError("tokenizer.vocab_size must be between 261 and 65534")

    for key in ("row_len", "workers"):
        _integer("packing", key, packing[key])

    for key in ("depth", "aspect_ratio", "head_dim", "max_seq_len"):
        _integer("model", key, model[key])
    _nonempty_string("model", "window_pattern", model["window_pattern"])
    if any(char not in "SL" for char in model["window_pattern"].upper()):
        raise ValueError("model.window_pattern must contain only S or L")
    if model["max_seq_len"] > packing["row_len"]:
        raise ValueError("model.max_seq_len cannot exceed packing.row_len")

    _nonempty_string("training", "model_tag", training["model_tag"])
    if not isinstance(training["device_type"], str) or training["device_type"] not in ("", "cuda", "cpu", "mps"):
        raise ValueError("training.device_type must be '', cuda, cpu or mps")

    # Match the native CLI: -1 means automatic device batch size.
    _integer("training", "device_batch_size", training["device_batch_size"], allow_auto=True)

    ratio = training["target_param_data_ratio"]
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("training.target_param_data_ratio must be a positive finite number")

    _integer("training", "warmup_steps", training["warmup_steps"], minimum=0)
    # Native -1 disables periodic evaluation and checkpoint saves (final save still happens).
    _integer("training", "eval_every", training["eval_every"], minimum=0, allow_auto=True)
    _integer("training", "eval_tokens", training["eval_tokens"], minimum=0)
    _integer("training", "save_every", training["save_every"], minimum=0, allow_auto=True)
    if training["eval_every"] > 0 and training["eval_tokens"] == 0:
        raise ValueError("training.eval_tokens must be positive when periodic evaluation is enabled")

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
