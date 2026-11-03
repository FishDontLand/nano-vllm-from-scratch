import torch
from model import qwen3
import json
from pathlib import Path
import safetensors
from transformers import AutoTokenizer

DTYPE_MAP = {
   'bfloat16': torch.bfloat16,
   'float32': torch.float32
}

def build_model_from_config(config_path: str) -> qwen3.Qwen3ForCausalLM:
  with open(config_path, 'r') as f:
    model_config = json.load(f)

  model = qwen3.Qwen3ForCausalLM(
    model_config['vocab_size'],
    model_config['num_hidden_layers'],
    model_config['hidden_size'],
    model_config['intermediate_size'],
    model_config['num_attention_heads'],
    model_config['num_key_value_heads'],
    model_config['head_dim'],
    model_config['rms_norm_eps'],
    model_config['rope_theta'],
    model_config['max_position_embeddings'],
    model_config['tie_word_embeddings']
  )

  dtype = DTYPE_MAP[model_config['torch_dtype']]
  model.to(dtype)

  return model



def load_weights(model: qwen3.Qwen3ForCausalLM, checkpoint_dir:Path) -> None:
    weight = safetensors.safe_open(
      filename=checkpoint_dir / 'model.safetensors',
      framework='pt',
      device='cpu'
    )
    model.load_state_dict(weight.get_tensors(), strict=False)

