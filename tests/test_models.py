import torch
import model.qwen3 as qwen3
import pytest

class TestQwen3Model:
  @torch.no_grad()
  def test_qwen3_model_three_layers(self):
      model = qwen3.Qwen3Model(
          vocab_size=8,
          num_hidden_layers=3,
          hidden_size=4,
          intermediate_size=6,
          num_heads=4,
          num_kv_heads=2,
          head_dim=8,
          rms_norm_eps=1e-6,
          rope_theta=10000.0,
          max_seq_len=8,
      ).to(device="cuda", dtype=torch.float16).eval()

      def fill_weight(module, period, divisor, shift=0):
          weight = module.weight
          values = torch.arange(
              weight.numel(), device="cuda", dtype=torch.float32
          ).reshape(weight.shape)
          weight.copy_(
              ((values + shift) % period - period // 2) / divisor
          )

      fill_weight(model.embed_tokens, period=17, divisor=8)

      projection_specs = [
          ("self_attn.q_proj", 11, 8),
          ("self_attn.k_proj", 7, 4),
          ("self_attn.v_proj", 9, 8),
          ("self_attn.o_proj", 13, 16),
          ("mlp.gate_proj", 7, 8),
          ("mlp.up_proj", 9, 8),
          ("mlp.down_proj", 11, 16),
      ]
      norm_weights = {
          "input_layernorm": [1, 0.75, 1.25, 0.5],
          "post_attention_layernorm": [0.5, 1.5, 0.75, 1.25],
          "self_attn.q_norm": [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
          "self_attn.k_norm": [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
      }

      assert len(model.layers) == 3

      for index, layer in enumerate(model.layers):
          # Different weights in each decoder layer.
          for name, period, divisor in projection_specs:
              projection = layer.get_submodule(name)
              assert projection.bias is None
              fill_weight(projection, period, divisor, shift=3 * index)

          for name, values in norm_weights.items():
              scale = torch.tensor(values, device="cuda") + index / 8
              layer.get_submodule(name).weight.copy_(scale)

      model.norm.weight.copy_(torch.tensor(
          [1.25, 0.75, 1.5, 0.5], device="cuda"
      ))

      # Integer token IDs: [batch=2, sequence=3].
      input_ids = torch.tensor(
          [[0, 3, 6], [1, 5, 2]],
          device="cuda",
          dtype=torch.long,
      )
      positions = torch.tensor(
          [[0, 2, 7], [1, 4, 6]],
          device="cuda",
          dtype=torch.long,
      )

      # Final normalized hidden states: [2, 3, 4].
      expected = torch.tensor([
          [
              [-1.89686294, -0.95122904, 0.10454605, -0.14471088],
              [0.91346757, -0.37957929, 2.20292396, 0.51307654],
              [0.74817186, -0.53640114, 0.20539560, 0.88197023],
          ],
          [
              [-1.04565952, -1.20240758, 0.56260021, 0.38381654],
              [-1.41543408, -1.16160508, 0.40222095, 0.24853637],
              [0.53614213, -0.90402617, 1.51789164, 0.57860165],
          ],
      ], device="cuda", dtype=torch.float32)

      actual = model(input_ids, positions)

      assert actual.shape == (2, 3, 4)
      assert actual.dtype == torch.float16
      assert actual.device == input_ids.device

      torch.testing.assert_close(
          actual.float(), expected, rtol=5e-3, atol=2e-3
      )

class TestQwen3ForCasualLM:

  @pytest.fixture(params=[True, False], ids=["tied", "untied"])
  @torch.no_grad()
  def model_case(self, request):
      tied = request.param
      model = qwen3.Qwen3ForCausalLM(
          vocab_size=8,
          num_hidden_layers=3,
          hidden_size=4,
          intermediate_size=6,
          num_heads=4,
          num_kv_heads=2,
          head_dim=8,
          rms_norm_eps=1e-6,
          rope_theta=10000.0,
          max_seq_len=8,
          tie_word_embeddings=tied,
      ).to(device="cuda", dtype=torch.float16).eval()

      embedding = model.model.embed_tokens.weight
      output_weight = model.lm_head.weight

      if tied:
          # Must share the parameter, not merely have equal values.
          assert output_weight is embedding
      else:
          assert output_weight is not embedding
          assert output_weight.data_ptr() != embedding.data_ptr()

      assert model.lm_head.bias is None

      def fill_weight(module, period, divisor, shift=0):
          weight = module.weight
          values = torch.arange(
              weight.numel(), device="cuda", dtype=torch.float32
          ).reshape(weight.shape)
          weight.copy_(
              ((values + shift) % period - period // 2) / divisor
          )
  
      fill_weight(model.model.embed_tokens, 17, 8)

      # For tied weights, filling the embedding already fills lm_head.
      if not tied:
          fill_weight(model.lm_head, 19, 16)

      projection_specs = [
          ("self_attn.q_proj", 11, 8),
          ("self_attn.k_proj", 7, 4),
          ("self_attn.v_proj", 9, 8),
          ("self_attn.o_proj", 13, 16),
          ("mlp.gate_proj", 7, 8),
          ("mlp.up_proj", 9, 8),
          ("mlp.down_proj", 11, 16),
      ]
      norm_weights = {
          "input_layernorm": [1, 0.75, 1.25, 0.5],
          "post_attention_layernorm": [0.5, 1.5, 0.75, 1.25],
          "self_attn.q_norm": [1, 0.5, 1.5, 1, 0.75, 1.25, 0.5, 1.5],
          "self_attn.k_norm": [0.75, 1.5, 0.5, 1.25, 1, 0.5, 1.5, 1],
      }

      for index, layer in enumerate(model.model.layers):
          for name, period, divisor in projection_specs:
              projection = layer.get_submodule(name)
              assert projection.bias is None
              fill_weight(projection, period, divisor, shift=3 * index)

          for name, values in norm_weights.items():
              scale = torch.tensor(values, device="cuda") + index / 8
              layer.get_submodule(name).weight.copy_(scale)

      model.model.norm.weight.copy_(torch.tensor(
          [1.25, 0.75, 1.5, 0.5], device="cuda"
      ))

      return model, tied


  @torch.no_grad()
  def test_forward(self, model_case):
      model, _ = model_case

      input_ids = torch.tensor(
          [[0, 3, 6], [1, 5, 2]],
          device="cuda",
          dtype=torch.long,
      )
      positions = torch.tensor(
          [[0, 2, 7], [1, 4, 6]],
          device="cuda",
          dtype=torch.long,
      )

      # forward returns hidden states, before lm_head.
      # Identical backbone weights → identical results in both tying cases.
      expected = torch.tensor([
          [
              [-1.89686294, -0.95122904, 0.10454605, -0.14471088],
              [0.91346757, -0.37957929, 2.20292396, 0.51307654],
              [0.74817186, -0.53640114, 0.20539560, 0.88197023],
          ],
          [
              [-1.04565952, -1.20240758, 0.56260021, 0.38381654],
              [-1.41543408, -1.16160508, 0.40222095, 0.24853637],
              [0.53614213, -0.90402617, 1.51789164, 0.57860165],
          ],
      ], device="cuda", dtype=torch.float32)

      actual = model(input_ids, positions)

      assert actual.shape == (2, 3, 4)
      assert actual.dtype == torch.float16
      assert actual.device == input_ids.device

      torch.testing.assert_close(
          actual.float(), expected, rtol=5e-3, atol=2e-3
      )


  @torch.no_grad()
  def test_compute_logits(self, model_case):
      model, tied = model_case

      # Two independently specified hidden vectors: [batch=2, hidden_size=4].
      hidden_states = torch.tensor(
          [[1, 0.5, -1, 2], [-0.5, 1, 0, 0.5]],
          device="cuda",
          dtype=torch.float16,
      )

      if tied:
          expected_values = [
              [-1.9375, -0.6875, 0.5625, 1.8125,
              -0.125, -1.0, 0.25, 1.5],
              [-0.6875, -0.1875, 0.3125, 0.8125,
              -1.875, -0.3125, 0.1875, 0.6875],
          ]
      else:
          expected_values = [
              [-1.125, -0.5, 0.125, 0.75,
              -1.0, -0.96875, -0.34375, 0.28125],
              [-0.40625, -0.15625, 0.09375, 0.34375,
              0.0, -0.34375, -0.09375, 0.15625],
          ]

      expected = torch.tensor(
          expected_values, device="cuda", dtype=torch.float16
      )
      actual = model.compute_logits(hidden_states)

      # These particular values are exactly representable in FP16.
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)