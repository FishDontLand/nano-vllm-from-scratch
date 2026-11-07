import torch

from model.cache import KVCache
from model.layers import Qwen3Attention
from model.qwen3 import Qwen3ForCausalLM
from examples.generate import generate_sample
from utils import loader
from transformers import AutoTokenizer
from pathlib import Path

def test_append_twice():
    cache = KVCache(batch_size=1, capacity=4, num_kv_heads=2, head_dim=2)

    first_k = torch.tensor([[
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]])
    first_v = torch.tensor([[
        [[11.0, 12.0], [13.0, 14.0]],
        [[15.0, 16.0], [17.0, 18.0]],
    ]])

    actual_k, actual_v = cache.append(first_k, first_v)

    torch.testing.assert_close(actual_k, first_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, first_v, rtol=0, atol=0)
    assert cache.length == 2

    second_k = torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]])
    second_v = torch.tensor([[[[19.0, 20.0], [21.0, 22.0]]]])

    actual_k, actual_v = cache.append(second_k, second_v)

    expected_k = torch.tensor([[
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ]])
    expected_v = torch.tensor([[
        [[11.0, 12.0], [13.0, 14.0]],
        [[15.0, 16.0], [17.0, 18.0]],
        [[19.0, 20.0], [21.0, 22.0]],
    ]])

    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
    assert cache.length == 3

@torch.inference_mode()
def test_cached_attention_matches_full_sequence():
    torch.manual_seed(0)

    attention = Qwen3Attention(
        hidden_size=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=8,
    ).to(device="cuda", dtype=torch.float16).eval()

    cache = KVCache(
        batch_size=1, capacity=4, num_kv_heads=1, head_dim=8
    ).to(device="cuda", dtype=torch.float16)

    x = torch.randn(1, 4, 16, device="cuda", dtype=torch.float16)
    positions = torch.arange(4, device="cuda").unsqueeze(0)

    # Expected: process all four tokens together.
    expected = attention(x, positions, cache=None)[:, 3:]

    # Actual: process three tokens, then one token using the same cache.
    attention(x[:, :3], positions[:, :3], cache=cache)
    actual = attention(x[:, 3:], positions[:, 3:], cache=cache)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-3)

@torch.inference_mode()
def test_cached_qwen_matches_full_sequence():
    torch.manual_seed(0)

    model = Qwen3ForCausalLM(
        vocab_size=16,
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=8,
        tie_word_embeddings=True,
    ).to(device="cuda", dtype=torch.float16).eval()

    caches = [
        KVCache(batch_size=1, capacity=4, num_kv_heads=1, head_dim=8)
        .to(device="cuda", dtype=torch.float16)
        for _ in model.model.layers
    ]
    input_ids = torch.tensor([[1, 3, 5, 7]], device="cuda")
    positions = torch.arange(4, device="cuda").unsqueeze(0)

    # Expected: the fourth token's output when processing all tokens together.
    expected = model(input_ids, positions)[:, 3:]

    # Actual: process three tokens, then the fourth using the same caches.
    model(input_ids[:, :3], positions[:, :3], caches=caches)
    actual = model(input_ids[:, 3:], positions[:, 3:], caches=caches)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-3)


def test_cache():
    model = loader.build_model_from_config('../checkpoints/Qwen3-0.6B/config.json')
    loader.load_weights(model, Path('../checkpoints/Qwen3-0.6B'))
    model.to('cuda:0')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("../checkpoints/Qwen3-0.6B", use_fast=True, local_files_only=True)
    
    q = "what is the capital city of us?"
    results1 = generate_sample(q, max_new_tokens=2, model=model, tokenizer=tokenizer)
    results2 = generate_sample(q, max_new_tokens=2, model=model, tokenizer=tokenizer, use_cache=True)
    assert results1 == results2
