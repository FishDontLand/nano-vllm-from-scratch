import torch
import model.utils as utils
from torch.nn.attention import SDPBackend, sdpa_kernel
import pytest
import torch.nn.functional as F

def test_dense_attention():
    # [batch=1, query_heads=4, queries=2, head_dim=2]
    q = torch.tensor([[
        [[1., 0.], [0., 1.]],
        [[0., 1.], [1., 1.]],
        [[1., -1.], [2., 0.]],
        [[-1., 1.], [0., -1.]],
    ]])

    # [batch=1, kv_heads=2, keys=4, head_dim=2]
    k = torch.tensor([[
        [[1., 0.], [0., 1.], [1., 1.], [9., 9.]],
        [[-1., 0.], [0., 2.], [1., -1.], [9., 9.]],
    ]])

    v = torch.tensor([[
        [[1., 2.], [3., 0.], [-1., 4.], [100., 100.]],
        [[0., 1.], [2., 3.], [4., -2.], [100., 100.]],
    ]])

    # First query sees keys 0–1; second query sees keys 0–2.
    # Key 3 is padding.
    query_positions = torch.tensor([[1, 2]])
    key_lengths = torch.tensor([3])

    expected = torch.tensor([[
        [[1.66047690, 1.33952310], [1.00000000, 2.00000000]],
        [[2.33952310, 0.66047690], [0.48953047, 2.51046953]],
        [[0.66047690, 1.66047690], [3.44505915, -0.93036641]],
        [[1.33952310, 2.33952310], [2.62857973, -0.71131609]],
    ]])

    with torch.no_grad():
        actual = utils.dense_attention(
            q, k, v, query_positions, key_lengths
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch,hq,hkv,q_len,k_len,dim",
    [
        (4, 8, 2, 64, 256, 64),    # GQA, multiple query tokens
        (8, 12, 3, 1, 512, 128),   # GQA, single-token decoding
        (3, 8, 8, 128, 128, 64),   # Standard MHA, full prompts
    ],
)


def test_dense_attention_large(batch, hq, hkv, q_len, k_len, dim):
    rng = torch.Generator(device="cpu").manual_seed(42)

    def random_tensor(*shape):
        return torch.randn(
            *shape, generator=rng, dtype=torch.float32, device="cpu"
        )

    # Transposing also exercises noncontiguous inputs.
    q = random_tensor(batch, q_len, hq, dim).transpose(1, 2)
    k = random_tensor(batch, k_len, hkv, dim).transpose(1, 2)
    v = random_tensor(batch, k_len, hkv, dim).transpose(1, 2)

    # Different valid lengths across the batch.
    key_lengths = torch.linspace(q_len, k_len, batch).long()

    # Queries represent the final q_len tokens of each valid sequence.
    query_positions = (
        key_lengths[:, None] - q_len + torch.arange(q_len)[None, :]
    )

    # PyTorch's boolean attention mask uses True for visible keys.
    key_indices = torch.arange(k_len)[None, None, :]
    visible = (
        (key_indices <= query_positions[:, :, None])
        & (key_indices < key_lengths[:, None, None])
    ).unsqueeze(1)

    # Expand KV heads for the CPU reference.
    reference_k = k.repeat_interleave(hq // hkv, dim=1)
    reference_v = v.repeat_interleave(hq // hkv, dim=1)

    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        expected = F.scaled_dot_product_attention(
            q.double(),
            reference_k.double(),
            reference_v.double(),
            attn_mask=visible,
            dropout_p=0.0,
            is_causal=False,  # Absolute-position mask already supplied.
        ).float()

        actual = utils.dense_attention(
            q, k, v, query_positions, key_lengths
        )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)