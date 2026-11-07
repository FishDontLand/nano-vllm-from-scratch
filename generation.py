import torch
import torch.nn as nn
from model.cache import KVCache 

def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    max_seq_len: int,
    use_cache: bool = False
) -> list[list[int]]:
  if input_ids.size()[-1] >= max_seq_len:
    return input_ids.cpu().tolist()

  if input_ids.size()[-1] + max_new_tokens > max_seq_len:
     raise ValueError(f"Generated more tokens than capacity!")

  init_seq_len = input_ids.size()[1]
  batch_size = input_ids.size()[0]

  eos_positions = torch.tensor(max_new_tokens - 1, dtype=torch.long, device=input_ids.device).repeat(batch_size)
  has_eos_reached = torch.tensor(False, device=input_ids.device).repeat(batch_size)

  if use_cache:
    kv_caches = []
    for _ in range(model.num_hidden_layers):
      cache = KVCache(
        input_ids.size()[0],
        max_seq_len,
        model.num_kv_heads,
        model.head_dim
        )
      cache.to(input_ids.device)
      kv_caches.append(cache)
  else:
    kv_caches = None

  with torch.inference_mode():
    positions = torch.arange(0, input_ids.size()[-1])
    positions = positions.to(input_ids.device)
    positions = positions.repeat(batch_size).view(*input_ids.size())
            
    for i in range(max_new_tokens):
      if use_cache and i > 0:
        embeddings = model(max_arg, positions=positions, caches=kv_caches)
      else:
        embeddings = model(input_ids, positions=positions, caches=kv_caches)
      last_embedding = embeddings[:, [-1], :]
      logits = model.compute_logits(last_embedding)
      max_arg = torch.argmax(logits, dim=-1)
      input_ids = torch.concat([input_ids, max_arg], dim=-1)
      if use_cache:
        positions = torch.tensor(init_seq_len + i, device=input_ids.device, dtype=torch.long).repeat(batch_size, 1)
      else:
        positions = torch.concat([positions,  torch.tensor(init_seq_len + i, device=input_ids.device, dtype=torch.long).repeat(batch_size, 1)], dim=-1)

      is_eos_reached = (max_arg.view(batch_size) == eos_token_id)
      has_eos_reached = has_eos_reached | is_eos_reached
      eos_positions = torch.where(is_eos_reached & (i < eos_positions), i, eos_positions)

      if has_eos_reached.all().item():
        break

    eos_positions = eos_positions.tolist()
    output_ids = input_ids[:, init_seq_len:].tolist()
    outputs = []
    for output, end_position in zip(output_ids, eos_positions):
       outputs.append(output[:(end_position + 1)])
    return outputs
  



