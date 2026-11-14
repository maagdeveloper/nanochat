
"""Export a native nanochat checkpoint to Hugging Face."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import PreTrainedTokenizerFast
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from consumer.artifacts import plan_artifacts
from consumer.config import load_config
from consumer.hf.configuration_nanochat_refactor import NanoChatRefactorConfig
from consumer.hf.modeling_nanochat_refactor import NanoChatRefactorForCausalLM
from harness.checkpoint import get_checkpoint_dir
from nanochat.tokenizer import RustBPETokenizer
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Export nanochat to Hugging Face"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Native nanochat model_*.pt checkpoint",
    )

    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Native tokenizer directory containing tokenizer.pkl",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    explicit = args.checkpoint is not None

    if explicit != (args.tokenizer_dir is not None):
        parser.error(
            "--checkpoint and --tokenizer-dir must be provided together"
        )

    if explicit:
        if args.output is None:
            parser.error("--output is required with --checkpoint")

        checkpoint = args.checkpoint.expanduser().resolve()
        tokenizer_dir = args.tokenizer_dir.expanduser().resolve()

    else:
        cfg = load_config()
        model_tag = cfg["training"]["model_tag"]

        checkpoint_dir = Path(
            get_checkpoint_dir(model_tag, "base")
        )

        checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))

        if not checkpoints:
            parser.error("No native checkpoint found")

        checkpoint = checkpoints[-1]

        tokenizer_dir = (
            Path(plan_artifacts(cfg)["tokenizer"]) / "tokenizer"
        )

    # Each checkpoint must have matching native metadata.
    step = checkpoint.stem.removeprefix("model_")

    if not checkpoint.name.startswith("model_") or not step.isdigit():
        parser.error("Expected checkpoint named model_<step>.pt")

    metadata_path = checkpoint.with_name(f"meta_{step}.json")

    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    if not metadata_path.is_file():
        parser.error(f"Metadata not found: {metadata_path}")

    if not (tokenizer_dir / "tokenizer.pkl").is_file():
        parser.error(f"Tokenizer not found: {tokenizer_dir}")

    metadata = json.loads(metadata_path.read_text())

    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint.parent.parent / "hf-base"
    )


    print("Checkpoint:", checkpoint)
    print("Tokenizer:", tokenizer_dir)
    print("Output:", output_dir)

    if args.dry_run:
        print("DRY RUN: no files written")
        return

    if output_dir.exists() or output_dir.is_symlink():
        parser.error(f"Export already exists: {output_dir}")

    native_tokenizer = RustBPETokenizer.from_directory(
        str(tokenizer_dir)
    )

    model_config = metadata["model_config"]

    assert (
        native_tokenizer.get_vocab_size()
        == model_config["vocab_size"]
    )

    NanoChatRefactorConfig.register_for_auto_class("AutoConfig")
    NanoChatRefactorForCausalLM.register_for_auto_class(
        "AutoModelForCausalLM"
    )

    bos_id = native_tokenizer.get_bos_token_id()

    hf_config = NanoChatRefactorConfig(
        **model_config,
        bos_token_id=bos_id,
        pad_token_id=bos_id,
    )

    model = NanoChatRefactorForCausalLM(hf_config)

    weights = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    # Match nanochat.engine stopping semantics.
    model.generation_config.eos_token_id = [
        bos_id,
        native_tokenizer.encode_special("<|assistant_end|>"),
    ]

    model.generation_config.pad_token_id = bos_id

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Write everything before publishing the export directory.
    with TemporaryDirectory(
        dir=output_dir.parent,
        prefix=".hf-export-",
    ) as temporary:
        temporary = Path(temporary)

        model.save_pretrained(
            temporary,
            safe_serialization=True,
        )

        convert_tiktoken_to_fast(
            native_tokenizer.enc,
            str(temporary),
        )

        # Preserve native special tokens and their vocabulary IDs.
        special_tokens = sorted(
            native_tokenizer.get_special_tokens(),
            key=native_tokenizer.encode_special,
        )

        bos_token = next(
            token for token in special_tokens
            if native_tokenizer.encode_special(token) == bos_id
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(temporary / "tokenizer.json"),
            bos_token=bos_token,
            pad_token=bos_token,
            padding_side="left",
            additional_special_tokens=[
                token for token in special_tokens
                if token != bos_token
            ],
            clean_up_tokenization_spaces=False,
        )

        # Verify that conversion preserved the native token IDs.
        assert len(tokenizer) == native_tokenizer.get_vocab_size()

        for token in special_tokens:
            assert (
                    tokenizer.convert_tokens_to_ids(token)
                    == native_tokenizer.encode_special(token)
            )

        tokenizer.save_pretrained(temporary)

        # Remove intermediate files produced by the converter.
        intermediate = temporary / "tiktoken"
        if intermediate.is_dir():
            shutil.rmtree(intermediate)

        temporary.rename(output_dir)

    print("Export completed:", output_dir)


if __name__ == "__main__":
    main()