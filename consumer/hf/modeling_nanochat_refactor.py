
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache

from .configuration_nanochat_refactor import NanoChatRefactorConfig


def has_ve(layer_idx, n_layer):
    """Match nanochat.gpt.has_ve."""
    return layer_idx % 2 == (n_layer - 1) % 2

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotate(x, cos, sin):
    # nanochat uses half-split rotation with negative theta.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    return torch.cat((
        x1 * cos + x2 * sin,
        -x1 * sin + x2 * cos,
    ), dim=-1)

class NanoChatAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        dim = config.n_embd
        head_dim = dim // config.n_head
        kv_dim = config.n_kv_head * head_dim

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(
                config.ve_gate_channels,
                config.n_kv_head,
                bias=False,
            )


class NanoChatMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.n_embd
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)


class NanoChatBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMLP(config)


class NanoChatRefactorForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatRefactorConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)

        dim = config.n_embd
        vocab = ((config.vocab_size + 63) // 64) * 64
        kv_dim = config.n_kv_head * (dim // config.n_head)

        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, dim)
        self.transformer.h = nn.ModuleList([
            NanoChatBlock(config, i)
            for i in range(config.n_layer)
        ])

        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(vocab, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })

        self.resid_lambdas = nn.Parameter(
            torch.linspace(1.15, 1.05, config.n_layer)
        )
        self.x0_lambdas = nn.Parameter(
            torch.linspace(0.20, 0.05, config.n_layer)
        )
        self.backout_lambda = nn.Parameter(
            torch.full((1,), 0.2)
        )

        self.lm_head = nn.Linear(dim, vocab, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, embeddings):
        self.transformer.wte = embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, embeddings):
        self.lm_head = embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            past_len = past_key_values.get_seq_length()
        else:
            if past_key_values is not None:
                raise ValueError("past_key_values requires use_cache=True")

            past_len = 0



        has_padding = False

        if attention_mask is not None:
            expected_length = past_len + input_ids.shape[1]

            if attention_mask.shape != (
                input_ids.shape[0], expected_length
            ):
                raise ValueError("Invalid attention_mask shape")

            if not torch.all(
                (attention_mask == 0) | (attention_mask == 1)
            ).item():
                raise ValueError("attention_mask must contain 0 or 1")

            # For now, support left padding only: 000111.
            if torch.any(
                attention_mask[:, 1:] < attention_mask[:, :-1]
            ).item():
                raise ValueError("Only left padding is supported")

            if not torch.all(attention_mask.sum(-1) > 0).item():
                raise ValueError("Each sequence needs a real token")

            has_padding = torch.any(
                attention_mask == 0
            ).item()

            if not has_padding:
                attention_mask = None

        if labels is not None:
            raise NotImplementedError("Training loss is not implemented yet")

        c = self.config
        B, T = input_ids.shape

        if c.window_pattern != "L":
            raise NotImplementedError("Only full causal attention is supported for now")

        if past_len + T > c.sequence_len:
            raise ValueError("Input exceeds the configured context length")

        head_dim = c.n_embd // c.n_head

        # Match nanochat's computation dtype on our RTX 3060.
        dtype = torch.bfloat16 if input_ids.is_cuda else torch.float32

        # RoPE: same frequencies, layout and rotation as nanochat.
        channels = torch.arange(
            0, head_dim, 2,
            dtype=torch.float32,
            device=input_ids.device,
        )

        positions = torch.arange(
            past_len,
            past_len + T,
            dtype=torch.float32,
            device=input_ids.device,
        )

        inv_freq = 1.0 / (c.rope_theta ** (channels / head_dim))

        if has_padding:
            # Each sequence starts its real token positions at zero.
            position_ids = (
                attention_mask.long().cumsum(-1) - 1
            ).clamp_min(0)[:, -T:]

            freqs = (
                position_ids.float().unsqueeze(-1)
                * inv_freq
            )

            cos = freqs.cos().to(dtype)[:, :, None, :]
            sin = freqs.sin().to(dtype)[:, :, None, :]

        else:
            # Preserve the exact original nanochat path.
            freqs = torch.outer(positions, inv_freq)

            cos = freqs.cos().to(dtype)[None, :, None, :]
            sin = freqs.sin().to(dtype)[None, :, None, :]

        # Native nanochat keeps master weights in fp32,
        # and explicitly casts linear weights to activation dtype.
        def linear(x, layer):
            return F.linear(x, layer.weight.to(x.dtype))

        x = self.transformer.wte(input_ids).to(dtype)
        x = norm(x)
        x0 = x
        x_backout = None

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            xn = norm(x)

            q = linear(xn, block.attn.c_q).view(
                B, T, c.n_head, head_dim
            )
            k = linear(xn, block.attn.c_k).view(
                B, T, c.n_kv_head, head_dim
            )
            v = linear(xn, block.attn.c_v).view(
                B, T, c.n_kv_head, head_dim
            )

            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](input_ids)
                ve = ve.to(x.dtype).view(
                    B, T, c.n_kv_head, head_dim
                )

                gate = 3 * torch.sigmoid(
                    linear(xn[..., :c.ve_gate_channels], block.attn.ve_gate)
                )

                v = v + gate.unsqueeze(-1) * ve

            q = norm(rotate(q, cos, sin)) * 1.2
            k = norm(rotate(k, cos, sin)) * 1.2

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if use_cache:
                k, v = past_key_values.update(k, v, i)

            if has_padding:
                query_positions = past_len + torch.arange(
                    T, device=input_ids.device
                )
                key_positions = torch.arange(
                    k.size(-2), device=input_ids.device
                )

                causal = (
                    key_positions[None, :]
                    <= query_positions[:, None]
                )

                valid_keys = attention_mask[
                    :, None, None, :k.size(-2)
                ].bool()

                attn_mask = (
                    valid_keys
                    & causal[None, None, :, :]
                )

                # Avoid fully masked rows for padding queries.
                invalid_queries = ~attention_mask[
                    :, -T:
                ].bool()

                diagonal = (
                    key_positions[None, :]
                    == query_positions[:, None]
                )

                attn_mask |= (
                    invalid_queries[:, None, :, None]
                    & diagonal[None, None, :, :]
                )

                is_causal = False

            elif past_len == 0:
                attn_mask = None
                is_causal = True

            elif T == 1:
                attn_mask = None
                is_causal = False

            else:
                query_positions = past_len + torch.arange(
                    T, device=input_ids.device
                )
                key_positions = torch.arange(
                    k.size(-2), device=input_ids.device
                )

                attn_mask = (
                    key_positions[None, :]
                    <= query_positions[:, None]
                )
                is_causal = False

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=(c.n_head != c.n_kv_head),
            )

            y = y.transpose(1, 2).contiguous().view(B, T, -1)
            x = x + linear(y, block.attn.c_proj)

            h = linear(norm(x), block.mlp.c_fc)
            h = F.relu(h).square()
            x = x + linear(h, block.mlp.c_proj)

            if i == c.n_layer // 2:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout

        x = norm(x)

        logits = linear(x, self.lm_head)[..., :c.vocab_size]
        logits = logits.float()
        logits = c.final_logit_softcapping * torch.tanh(
            logits / c.final_logit_softcapping
        )

        if not return_dict:
            return (logits, past_key_values) if use_cache else (logits,)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )
