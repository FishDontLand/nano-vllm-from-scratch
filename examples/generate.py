import torch
from pathlib import Path
from utils import loader
from transformers import AutoTokenizer
import generation
import argparse

def generate_sample(input: str, max_new_tokens=32, use_cache=False, decode=True, model=None, tokenizer=None):
  if model is None:
    model = loader.build_model_from_config('../checkpoints/Qwen3-0.6B/config.json')
    loader.load_weights(model, Path('../checkpoints/Qwen3-0.6B'))
    model.to('cuda:0')
    model.eval()

  if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained("../checkpoints/Qwen3-0.6B", use_fast=True, local_files_only=True)
  input_ids = tokenizer(input)['input_ids']
  input_ids = torch.tensor([input_ids], dtype=torch.long, device='cuda:0')

  answer = generation.generate_greedy(
    model, 
    input_ids,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    max_seq_len=1024,
    use_cache=use_cache
  )

  if decode:
    return tokenizer.decode(answer[0])
  else:
    return answer[0]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="arguments for llm to generate answer")

  parser.add_argument("prompt", help="prompt from user")
  parser.add_argument("--maxToken", type=int)
  parser.add_argument("--useCache", action="store_true")
  parser.add_argument("--noDecode", action="store_true")

  args = parser.parse_args()

  if args.maxToken:
    answer = generate_sample(args.prompt, args.maxToken, True if args.useCache else False, False if args.noDecode else True)
  else:
    answer = generate_sample(args.prompt, use_cache=True if args.useCache else False, decode=False if args.noDecode else True)

  print(answer)


