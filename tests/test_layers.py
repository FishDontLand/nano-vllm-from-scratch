import pytest
import torch
from torch import nn

from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp


import model.layers as layers

class TestReplicatedLinear:
    @pytest.fixture(autouse=True)
    def single_process(self, monkeypatch):
        # Isolate these unit tests from distributed process-group setup.
        monkeypatch.setattr("model.layers.get_rank", lambda: 0)
        monkeypatch.setattr("model.layers.get_world_size", lambda: 1)

    @pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "with_bias"])
    def test_forward_with_and_without_bias(self, use_bias):
        inputs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
        bias = torch.tensor([0.5, -1.0])
        expected = torch.tensor([[-2.0, 4.0], [-3.0, -2.0]])

        layer = layers.ReplicatedLinear(3, 2, bias=use_bias)
        assert isinstance(layer.weight, nn.Parameter)
        with torch.no_grad():
            layer.weight.copy_(weight)
            if use_bias:
                layer.bias.copy_(bias)

        if use_bias:
            assert isinstance(layer.bias, nn.Parameter)
            assert "bias" in dict(layer.named_parameters())
        else:
            assert layer.bias is None

        torch.testing.assert_close(
            layer(inputs), expected + bias if use_bias else expected
        )

    def test_weight_loader_copies_values_and_rejects_wrong_shape(self):
        layer = layers.ReplicatedLinear(3, 2)
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        original_parameter = layer.weight

        layer.weight_loader(layer.weight, source)

        assert layer.weight is original_parameter
        assert layer.weight.requires_grad
        torch.testing.assert_close(layer.weight, source)
        source.add_(10)
        expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        torch.testing.assert_close(layer.weight, expected)

        # Same number of elements, but a different shape must be rejected.
        with pytest.raises(ValueError):
            layer.weight_loader(layer.weight, torch.zeros(3, 2))
        torch.testing.assert_close(layer.weight, expected)


class TestColumnParallelLinear:
    def _worker(self, rank, rendezvous):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

        dist.init_process_group(
            backend="gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )

        try:
            layer = layers.ColumnParallelLinear(
                input_size=2,
                output_size=4,
                bias=True,
            )

            weight = torch.tensor([
                [1., 2.],
                [3., 4.],
                [5., 6.],
                [7., 8.],
            ])
            bias = torch.tensor([10., 20., 30., 40.])

            # Test weight_loader: each rank gets its own rows.
            layer.weight_loader(layer.weight, weight)
            layer.weight_loader(layer.bias, bias)

            torch.testing.assert_close(
                layer.weight, weight.chunk(2, dim=0)[rank]
            )
            torch.testing.assert_close(
                layer.bias, bias.chunk(2, dim=0)[rank]
            )

            # Test forward: full expected output is [14, 30, 46, 62].
            x = torch.tensor([[2., 1.]])
            expected = [
                torch.tensor([[14., 30.]]),
                torch.tensor([[46., 62.]]),
            ]

            with torch.no_grad():
                actual = layer(x)

            torch.testing.assert_close(actual, expected[rank])

        finally:
            dist.destroy_process_group()

    def test_weight_loader_and_forward(self, tmp_path):
        rendezvous = (tmp_path / "group").resolve().as_uri()
        mp.spawn(self._worker, args=(rendezvous,), nprocs=2, join=True)

class TestMergedColumnParallelLinear:
    def _worker(self, rank, rendezvous):
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=3,
            timeout=timedelta(seconds=30),
        )

        try:
            layer = layers.MergedColumnParallelLinear(
                input_size=2,
                output_sizes=[6, 9],
                bias=False,
            ).to(device="cpu", dtype=torch.float32)

            first = torch.arange(12, dtype=torch.float32).reshape(6, 2)
            second = torch.arange(100, 118, dtype=torch.float32).reshape(9, 2)

            # Fill with NaN so missing writes fail the comparison.
            with torch.no_grad():
                layer.weight.fill_(float("nan"))

            layer.weight_loader(layer.weight, first, 0)
            layer.weight_loader(layer.weight, second, 1)

            # Each rank stores its first-projection shard,
            # followed by its second-projection shard.
            expected = [
                [[0, 1], [2, 3],
                [100, 101], [102, 103], [104, 105]],

                [[4, 5], [6, 7],
                [106, 107], [108, 109], [110, 111]],

                [[8, 9], [10, 11],
                [112, 113], [114, 115], [116, 117]],
            ]

            torch.testing.assert_close(
                layer.weight,
                torch.tensor(expected[rank], dtype=torch.float32),
                rtol=0,
                atol=0,
            )
        finally:
            dist.destroy_process_group()


    def test_merged_weight_loader(self, tmp_path):
        rendezvous = (tmp_path / "group").resolve().as_uri()
        mp.spawn(self._worker, args=(rendezvous,), nprocs=3, join=True)

class TestRMSNorm:

    @pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8)])
    @pytest.mark.parametrize("scale", [1.0, 1e-4])
    def test_rmsnorm_forward(self, shape, scale):
        torch.manual_seed(42)
        eps = 1e-6

        layer = layers.RMSNorm(hidden_size=8, eps=eps).to(
            device="cpu", dtype=torch.float32
        )
        reference = torch.nn.RMSNorm(8, eps=eps, dtype=torch.float32)

        # Nonuniform weights ensure forward actually applies the gain.
        weight = torch.linspace(0.5, 2.0, 8)
        x = torch.randn(shape) * scale

        with torch.no_grad():
            layer.weight.copy_(weight)
            reference.weight.copy_(weight)

            actual = layer(x)
            expected = reference(x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

class TestSwiGLU:
    def test_swiglu_forward(self):
        layer = layers.SwiGLU(hidden_size=2, intermediate_size=3).to(
            device="cpu", dtype=torch.float32
        )

        gate_weight = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ])
        up_weight = torch.tensor([
            [0.5, 1.0],
            [-1.0, 0.5],
            [1.0, 1.0],
        ])
        down_weight = torch.tensor([
            [1.0, -0.5, 2.0],
            [-1.0, 1.0, 0.25],
        ])

        # Shape: [batch=2, sequence=2, hidden_size=2].
        x = torch.tensor([
            [[1.0, 2.0], [-1.0, 0.5]],
            [[0.0, 0.0], [2.0, -1.0]],
        ])

        expected = torch.tensor([
            [[0.21399792, -2.02935251],
            [0.07911974, 0.42324187]],

            [[0.00000000, 0.00000000],
            [5.37926798, 1.38678415]],
        ])

        with torch.no_grad():
            layer.gate_proj.weight.copy_(gate_weight)
            layer.up_proj.weight.copy_(up_weight)
            layer.down_proj.weight.copy_(down_weight)

            actual = layer(x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class TestRotaryEmbedding:
    def test_rope_forward(self):
        rope = layers.RotaryEmbedding(
            head_dim=8,
            theta=10000.0,
            max_seq_len=8,
        )

        # Base values: [batch, sequence, head_dim].
        base_x = torch.tensor([
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [-2, 1, 0, 3, 4, -1, 2, -3],
            ],
            [
                [2, -1, 4, 0, 3, 5, -2, 1],
                [0, 3, -2, 1, -4, 2, 5, -1],
            ],
        ], dtype=torch.float32)

        # Three heads → [batch, sequence, heads, head_dim].
        x = torch.stack((base_x, -base_x, 2 * base_x), dim=2)

        positions = torch.tensor([
            [0, 1],
            [2, 7],
        ], dtype=torch.long)

        # Hard-coded split-halves RoPE results for base_x.
        base_expected = torch.tensor([
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [
                    -4.44648855, 1.09483758, -0.01999967, 3.00299850,
                    0.47826725, -0.89517075, 1.99990000, -2.99699850,
                ],
            ],
            [
                [
                    -3.56018595, -1.97341323, 4.03919736, -0.00200000,
                    0.57015434, 4.70166356, -1.91960535, 0.99999800,
                ],
                [
                    2.62794639, 1.00609119, -2.34481624, 1.00697544,
                    -3.01560902, 3.46233744, 4.84786931, -0.99297556,
                ],
            ],
        ], dtype=torch.float32)

        # Rotation is linear: negating/scaling inputs does the same to outputs.
        expected = torch.stack(
            (base_expected, -base_expected, 2 * base_expected),
            dim=2,
        )

        actual = rope(x, positions)

        assert actual.shape == (2, 2, 3, 8)
        torch.testing.assert_close(
            actual, expected, rtol=1e-5, atol=1e-6
        )

class TestQwen3Attention:
    @torch.no_grad()
    def test_qwen3_attention_forward(self):
        model = layers.Qwen3Attention(
            hidden_size=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_seq_len=8,
        ).to(device="cuda", dtype=torch.float16).eval()

        # Fixed projection weights.
        for name, period, divisor in [
            ("q_proj", 11, 8),
            ("k_proj", 7, 4),
            ("v_proj", 9, 8),
            ("o_proj", 13, 16),
        ]:
            weight = getattr(model, name).weight
            values = torch.arange(
                weight.numel(), device="cuda", dtype=torch.float32
            ).reshape(weight.shape)
            weight.copy_((values % period - period // 2) / divisor)

        model.q_norm.weight.copy_(torch.tensor(
            [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
            device="cuda",
        ))
        model.k_norm.weight.copy_(torch.tensor(
            [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
            device="cuda",
        ))

        # Input: [batch=2, sequence=3, hidden_size=4].
        # Internal Q:   [2, 3, 4, 8].
        # Internal K/V: [2, 3, 2, 8].
        # Sequence length differs from both head counts.
        x = torch.tensor([
            [
                [1, 0.5, -1, 2],
                [-0.5, 1, 0, 0.5],
                [2, -1, 0.5, 1],
            ],
            [
                [0, 1, 2, -1],
                [1, -0.5, 1.5, 0],
                [-1, 2, 0.5, -0.5],
            ],
        ], device="cuda", dtype=torch.float16)

        positions = torch.tensor(
            [[0, 1, 2], [0, 1, 2]],
            device="cuda",
            dtype=torch.long,
        )

        expected = torch.tensor([
            [
                [0.03515625, -0.28906250, -0.05468750, -0.32812500],
                [-0.01928438, -0.19558703, 0.12059359, -0.10327039],
                [0.06103483, -0.04004245, -0.10711959, -0.15269338],
            ],
            [
                [0.16406250, 0.87500000, -0.44531250, 0.46875000],
                [0.14316111, 0.56256465, -0.36574069, 0.20579324],
                [0.11639968, 0.37261566, -0.28395807, 0.11611802],
            ],
        ], device="cuda", dtype=torch.float32)

        actual = model(x, positions)

        assert actual.shape == x.shape
        assert actual.dtype == x.dtype
        assert actual.device == x.device

        torch.testing.assert_close(
            actual.float(), expected, rtol=5e-3, atol=1e-3
        )

    @torch.no_grad()
    def test_qwen3_attention_nonconsecutive_positions(self):
        model = layers.Qwen3Attention(
            hidden_size=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_seq_len=8,
        ).to(device="cuda", dtype=torch.float16).eval()

        # Same fixed weights as the previous test.
        for name, period, divisor in [
            ("q_proj", 11, 8),
            ("k_proj", 7, 4),
            ("v_proj", 9, 8),
            ("o_proj", 13, 16),
        ]:
            weight = getattr(model, name).weight
            values = torch.arange(
                weight.numel(), device="cuda", dtype=torch.float32
            ).reshape(weight.shape)
            weight.copy_((values % period - period // 2) / divisor)

        model.q_norm.weight.copy_(torch.tensor(
            [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
            device="cuda",
        ))
        model.k_norm.weight.copy_(torch.tensor(
            [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
            device="cuda",
        ))

        # [batch=2, selected_tokens=3, hidden_size=4]
        x = torch.tensor([
            [
                [1, 0.5, -1, 2],
                [-0.5, 1, 0, 0.5],
                [2, -1, 0.5, 1],
            ],
            [
                [0, 1, 2, -1],
                [1, -0.5, 1.5, 0],
                [-1, 2, 0.5, -0.5],
            ],
        ], device="cuda", dtype=torch.float16)

        positions = torch.tensor(
            [[0, 2, 7], [1, 4, 6]],
            device="cuda",
            dtype=torch.long,
        )

        expected = torch.tensor([
            [
                [0.03515625, -0.28906250, -0.05468750, -0.32812500],
                [-0.10315385, -0.14910490, 0.07418004, -0.04162461],
                [0.00639596, 0.08148117, -0.10122645, 0.02100350],
            ],
            [
                [0.16406250, 0.87500000, -0.44531250, 0.46875000],
                [0.12679410, 0.36276666, -0.36239524, 0.00469364],
                [0.18049804, 0.39503570, -0.19937132, 0.13897843],
            ],
        ], device="cuda", dtype=torch.float32)

        actual = model(x, positions)

        assert actual.shape == x.shape
        assert actual.dtype == x.dtype
        assert actual.device == x.device

        torch.testing.assert_close(
            actual.float(), expected, rtol=5e-3, atol=1e-3
        )

class TestQwen3DecoderLayer:
    @torch.no_grad()
    def test_qwen3_decoder_layer_forward(self):
        model = layers.Qwen3DecoderLayer(
            hidden_size=4,
            intermediate_size=6,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_seq_len=8,
        ).to(device="cuda", dtype=torch.float16).eval()

        # Fixed projection weights, independent of random initialization.
        for name, period, divisor in [
            ("self_attn.q_proj", 11, 8),
            ("self_attn.k_proj", 7, 4),
            ("self_attn.v_proj", 9, 8),
            ("self_attn.o_proj", 13, 16),
            ("mlp.gate_proj", 7, 8),
            ("mlp.up_proj", 9, 8),
            ("mlp.down_proj", 11, 16),
        ]:
            layer = model.get_submodule(name)
            assert layer.bias is None

            weight = layer.weight
            values = torch.arange(
                weight.numel(), device="cuda", dtype=torch.float32
            ).reshape(weight.shape)

            weight.copy_((values % period - period // 2) / divisor)

        # Distinct scales for all four normalization layers.
        norm_weights = {
            "input_layernorm": [1, 0.75, 1.25, 0.5],
            "post_attention_layernorm": [0.5, 1.5, 0.75, 1.25],
            "self_attn.q_norm": [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
            "self_attn.k_norm": [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
        }
        for name, values in norm_weights.items():
            model.get_submodule(name).weight.copy_(
                torch.tensor(values, device="cuda")
            )

        # [batch=2, sequence=3, hidden_size=4]
        x = torch.tensor([
            [
                [1, 0.5, -1, 2],
                [-0.5, 1, 0, 0.5],
                [2, -1, 0.5, 1],
            ],
            [
                [0, 1, 2, -1],
                [1, -0.5, 1.5, 0],
                [-1, 2, 0.5, -0.5],
            ],
        ], device="cuda", dtype=torch.float16)

        positions = torch.tensor(
            [[0, 2, 7], [1, 4, 6]],
            device="cuda",
            dtype=torch.long,
        )

        expected = torch.tensor([
            [
                [1.15200297, 0.46370480, -1.19163130, 1.71296432],
                [-0.58682733, 1.04850550, -0.01326120, 0.49836516],
                [2.26509785, -0.91891993, 0.53955713, 0.97286566],
            ],
            [
                [-0.18612900, 1.68832652, 1.58634905, -0.43460255],
                [0.81833256, 0.04127535, 0.92830377, 0.34733414],
                [-1.13994898, 2.52472776, 0.22734235, -0.06209362],
            ],
        ], device="cuda", dtype=torch.float32)

        actual = model(x, positions)

        assert actual.shape == x.shape
        assert actual.dtype == x.dtype
        assert actual.device == x.device

        # Allow FP16 rounding across the complete block.
        torch.testing.assert_close(
            actual.float(), expected, rtol=5e-3, atol=2e-3
        )