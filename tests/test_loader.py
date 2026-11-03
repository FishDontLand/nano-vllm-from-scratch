import torch
from utils import loader
from pathlib import Path
import safetensors

class TestLoader:
  @torch.no_grad()
  def test_loader(self):
    config_path = r'./checkpoints/Qwen3-0.6B/config.json'
    checkpoint_path = Path('./checkpoints/Qwen3-0.6B')

    model = loader.build_model_from_config(config_path)
    loader.load_weights(model, checkpoint_path)
    model_weights = model.state_dict()

    weights = safetensors.safe_open(
      filename= checkpoint_path / 'model.safetensors',
      framework='pt',
      device='cpu'
    )

    weights = weights.get_tensors()

    for k in weights:
      torch.testing.assert_close(model_weights[k], weights[k])

    for k in model_weights.keys():
      if k not in weights:
        assert 'rotary_emb' in k