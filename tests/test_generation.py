import pytest
import torch

from model.qwen3 import Qwen3ForCausalLM
from generation import generate_greedy


@pytest.fixture(scope="module")
@torch.no_grad()
def tiny_qwen():
    model = Qwen3ForCausalLM(
        vocab_size=8,
        num_hidden_layers=1,
        hidden_size=6,
        intermediate_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=8,
        tie_word_embeddings=False,
    ).to(device="cuda", dtype=torch.float16).eval()

    def fill_weight(module, period, divisor, shift=0):
        weight = module.weight
        values = torch.arange(
            weight.numel(), device="cuda", dtype=torch.float32
        ).reshape(weight.shape)
        weight.copy_(
            ((values + shift) % period - period // 2) / divisor
        )

    fill_weight(model.model.embed_tokens, 17, 8)
    fill_weight(model.lm_head, 19, 8, shift=11)

    layer = model.model.layers[0]

    for name, period, divisor in [
        ("self_attn.q_proj", 11, 8),
        ("self_attn.k_proj", 7, 4),
        ("self_attn.v_proj", 9, 8),
        ("self_attn.o_proj", 13, 16),
        ("mlp.gate_proj", 7, 8),
        ("mlp.up_proj", 9, 8),
        ("mlp.down_proj", 11, 16),
    ]:
        fill_weight(layer.get_submodule(name), period, divisor)

    norm_weights = {
        "input_layernorm": [1, 0.75, 1.25, 0.5, 1.5, 0.75],
        "post_attention_layernorm": [0.5, 1.5, 0.75, 1.25, 1, 0.5],
        "self_attn.q_norm": [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
        "self_attn.k_norm": [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
    }
    for name, values in norm_weights.items():
        layer.get_submodule(name).weight.copy_(
            torch.tensor(values, device="cuda")
        )

    model.model.norm.weight.copy_(torch.tensor(
        [1.25, 0.75, 1.5, 0.5, 1, 0.75],
        device="cuda",
    ))

    return model


@pytest.mark.parametrize("use_cache", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize(
    "max_new_tokens,eos_token_id,max_seq_len,expected",
    [
        # Both rows use the full token budget.
        (5, 6, 8, [[5, 7, 0, 5, 7], [0, 5, 7, 0, 5]]),
        # Row 2 reaches EOS immediately; row 1 continues until step 3.
        (5, 0, 8, [[5, 7, 0], [0]]),
        # Row 1 reaches EOS immediately; row 2 continues until step 2.
        (5, 5, 8, [[5], [0, 5]]),
        # Row 1 reaches its budget; row 2 reaches EOS.
        (2, 0, 8, [[5, 7], [0]]),
        # Two prompt tokens plus five output tokens exactly fill capacity.
        (5, 6, 7, [[5, 7, 0, 5, 7], [0, 5, 7, 0, 5]]),
        # Zero budget returns an empty result for each input row.
        (0, 6, 8, [[], []]),
    ],
)
def test_generate_greedy_with_qwen(
    tiny_qwen, use_cache, max_new_tokens, eos_token_id, max_seq_len, expected
):
    input_ids = torch.tensor(
        [[1, 3], [2, 4]], device="cuda", dtype=torch.long
    )
    original_input = input_ids.clone()

    actual = generate_greedy(
        model=tiny_qwen,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        max_seq_len=max_seq_len,
        use_cache=use_cache,
    )

    assert isinstance(actual, list)
    assert all(isinstance(row, list) for row in actual)
    assert all(isinstance(token, int) for row in actual for token in row)
    assert actual == expected

    torch.testing.assert_close(input_ids, original_input)


@pytest.mark.parametrize("use_cache", [False, True], ids=["uncached", "cached"])
def test_generate_greedy_rejects_length_overflow(tiny_qwen, use_cache):
    input_ids = torch.tensor(
        [[1, 3], [2, 4]], device="cuda", dtype=torch.long
    )

    # Each row has two prompt tokens and requests seven more: 9 > 8.
    with pytest.raises(ValueError):
        generate_greedy(
            model=tiny_qwen,
            input_ids=input_ids,
            max_new_tokens=7,
            eos_token_id=6,
            max_seq_len=8,
            use_cache=use_cache,
        )


@pytest.mark.parametrize("use_cache", [False, True], ids=["uncached", "cached"])
def test_generate_greedy_single_row_batch(tiny_qwen, use_cache):
    input_ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.long)

    actual = generate_greedy(
        model=tiny_qwen,
        input_ids=input_ids,
        max_new_tokens=5,
        eos_token_id=6,
        max_seq_len=8,
        use_cache=use_cache,
    )

    assert actual == [[5, 7, 0, 5, 7]]


