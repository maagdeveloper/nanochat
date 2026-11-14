
from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig


@strict
class NanoChatRefactorConfig(PreTrainedConfig):
    model_type = "nanochat_refactor"

    attribute_map = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_kv_head",
        "max_position_embeddings": "sequence_len",
    }

    # Native GPTConfig fields.
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"
    short_seq_len: int = 512

    # Native architecture constants.
    rope_theta: float = 100000.0
    ve_gate_channels: int = 12
    final_logit_softcapping: float = 15.0

    # Intended HF generation behavior.
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        if not self.window_pattern or any(
            c not in "SL" for c in self.window_pattern
        ):
            raise ValueError("window_pattern must contain only S or L")

        super().__post_init__(**kwargs)
