
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from consumer.config import DEFAULT_CONFIG, load_config
from consumer.artifacts import plan_artifacts, require_native_provenance
from nanochat.tokenizer import get_tokenizer_dir, tokenizer_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Consumer TOML configuration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-step", type=int, default=-1,
                        help="Resume a native checkpoint step (-1 = new run)")
    args = parser.parse_args()
    if args.resume_from_step != -1 and args.resume_from_step < 1:
        parser.error("--resume-from-step must be a positive checkpoint step")
    resuming = args.resume_from_step != -1

    cfg = load_config(args.config)
    # Verify raw inputs, tokenizer, packed payloads, and native preparation provenance.
    require_native_provenance(cfg)
    dataset = cfg["dataset"]
    packing = cfg["packing"]
    model = cfg["model"]
    training = cfg["training"]

    plan = plan_artifacts(cfg)
    active_dataset = plan["packed_name"]

    env = os.environ.copy()
    env["NANOCHAT_DATASET"] = active_dataset

    # One complete sequence per packed row.
    if model["max_seq_len"] != packing["row_len"]:
        parser.error("Model context must match packed row length for this run")

    # Check the actual prepared files, not just config values.
    from nanochat.dataloader import split_info

    os.environ["NANOCHAT_DATASET"] = active_dataset

    train_info = split_info("train", dataset["train_shards"])
    val_info = split_info("val")

    actual_tokenizer_id = tokenizer_id(get_tokenizer_dir())

    for name, info in (("train", train_info), ("val", val_info)):
        if info[3] != actual_tokenizer_id:
            parser.error(
                f"{name}: tokenizer ID mismatch: "
                f"packed={info[3]}, tokenizer={actual_tokenizer_id}"
            )

    for name, info in (("train", train_info), ("val", val_info)):
        row_len, _, vocab_size, _ = info

        if row_len != packing["row_len"]:
            parser.error(f"{name}: packed row length mismatch")
        if vocab_size != cfg["tokenizer"]["vocab_size"]:
            parser.error(f"{name}: packed vocabulary mismatch")

    command = [
        sys.executable, "-u", "-m", "scripts.base_train",
        "--run", "dummy",
        "--device-type", training["device_type"],
        "--depth", str(model["depth"]),
        "--aspect-ratio", str(model["aspect_ratio"]),
        "--head-dim", str(model["head_dim"]),
        "--max-seq-len", str(model["max_seq_len"]),
        "--window-pattern", model["window_pattern"],
        "--num-shards", str(dataset["train_shards"]),
        "--device-batch-size", str(training["device_batch_size"]),
        "--target-param-data-ratio", str(training["target_param_data_ratio"]),
        "--warmup-steps", str(training["warmup_steps"]),
        "--eval-every", str(training["eval_every"]),
        "--eval-tokens", str(training["eval_tokens"]),
        "--save-every", str(training["save_every"]),
        "--model-tag", training["model_tag"],
    ]

    if resuming:
        command.extend(["--resume-from-step", str(args.resume_from_step)])

    artifact_line = (
        f"artifact dataset={active_dataset} "
        f"tokenizer_id={actual_tokenizer_id}"
    )

    print(artifact_line, flush=True)

    print("Command:", shlex.join(command), flush=True)
    print(f"Active dataset: {active_dataset}", flush=True)
    print(f"Packed directory: {plan['packed']}", flush=True)
    print(f"Tokenizer ID: {actual_tokenizer_id}", flush=True)
    print(f"Train rows: {train_info[1]:,}", flush=True)
    print(f"Val rows: {val_info[1]:,}", flush=True)

    if args.dry_run:
        if resuming:
            print("DRY RUN: resume checkpoint not inspected")
        print("DRY RUN: no training performed")
        return

    from harness.checkpoint import get_checkpoint_dir

    log_dir = Path(get_checkpoint_dir(training["model_tag"], "base"))
    log_path = log_dir / "train.log"

    if resuming:
        step = args.resume_from_step
        required = [
            log_dir / f"model_{step:06d}.pt",
            log_dir / f"optim_{step:06d}_rank0.pt",
            log_dir / f"meta_{step:06d}.json",
            log_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error("Cannot resume; missing: " + ", ".join(missing))

        # Do not silently overwrite checkpoints newer than the requested step.
        if any(
            int(path.stem.split("_")[-1]) > step
            for path in log_dir.glob("model_*.pt")
            if path.stem.split("_")[-1].isdigit()
        ):
            parser.error("Newer checkpoints exist; refusing to rewind this run")

        # A consumer run records the dataset and native tokenizer identity.
        with log_path.open(encoding="utf-8") as log:
            first_line = log.readline().strip()
        if first_line != artifact_line:
            parser.error("Resume dataset/tokenizer differs from recorded run")

        metadata = json.loads((log_dir / f"meta_{step:06d}.json").read_text())
        saved_args = metadata.get("user_config", {})
        expected = {
            "depth": model["depth"],
            "aspect_ratio": model["aspect_ratio"],
            "head_dim": model["head_dim"],
            "max_seq_len": model["max_seq_len"],
            "window_pattern": model["window_pattern"],
            "num_shards": dataset["train_shards"],
            "device_batch_size": training["device_batch_size"],
            "target_param_data_ratio": training["target_param_data_ratio"],
            "warmup_steps": training["warmup_steps"],
            "model_tag": training["model_tag"],
        }
        if metadata.get("step") != step or any(
            saved_args.get(key) != value for key, value in expected.items()
        ):
            parser.error("Resume checkpoint configuration differs from this run")
    else:
        # Never silently overwrite logs or checkpoints for a new run.
        if log_path.exists() or any(log_dir.glob("model_*.pt")):
            parser.error(
                f"Run already exists: {log_dir}. "
                "Choose a new model_tag or resume explicitly."
            )
        log_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]

    with log_path.open("a" if resuming else "x", encoding="utf-8") as log:
        if resuming:
            log.write(f"\nresume_from_step={args.resume_from_step} {artifact_line}\n")
        else:
            log.write(artifact_line + "\n")
        log.flush()

        with subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)

            returncode = process.wait()

    if returncode:
        raise SystemExit(returncode)

    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
