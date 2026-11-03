import torch
from pathlib import Path
from utils import loader
from transformers import AutoTokenizer
import generation

def generate_sample(input: str):
  model = loader.build_model_from_config('../checkpoints/Qwen3-0.6B/config.json')
  loader.load_weights(model, Path('../checkpoints/Qwen3-0.6B'))
  model.to('cuda:0')
  model.eval()

  tokenizer = AutoTokenizer.from_pretrained("../checkpoints/Qwen3-0.6B", use_fast=True, local_files_only=True)
  input_ids = tokenizer(input)['input_ids']
  input_ids = torch.tensor([input_ids], dtype=torch.long, device='cuda:0')

  generation.generate_greedy(
    model, 
    input_ids,
  )
