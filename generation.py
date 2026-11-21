import torch
import torch.nn as nn
from model.cache import KVCache 
from model.batching import pack_prompts

def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    max_seq_len: int,
    use_cache: bool = False,
    return_embedding: bool = False
) -> tuple[list[list[int]], list[torch.Tensor]]:
  embedding_cache = []
  
  if input_ids.size()[-1] >= max_seq_len:
    return [], embedding_cache
  

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
        min(max_seq_len, input_ids.sie()[-1] + max_new_tokens),
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
      if return_embedding:
        embedding_cache.append(last_embedding)
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

    return outputs, embedding_cache
  
def generate_greedy_varlen(
    model: nn.Module,
    prompts: list[list[int]],
    max_new_tokens: int,
    eos_token_id: int,
    max_seq_len: int,
) -> list[list[int]]:
  filtered_prompts = [p for p in prompts if len(p) < max_seq_len]
  is_kept_idx = torch.tensor(
    [i for i in range(len(prompts)) if len(prompts[i]) < max_seq_len],
    dtype=torch.long,
    device=model.lm_head.weight.device
  )

  if len(filtered_prompts) == 0:
    raise ValueError("all prompts exceed the length capacity")

  total_len += sum([min(max_new_tokens + len(p), max_seq_len) for p in filtered_prompts])

  kv_caches = []
  for _ in range(model.num_hidden_layers):

    cache = KVCache(
      1,
      total_len,
      model.num_kv_heads,
      model.head_dim
    )
    
    cache.to(model.lm_head.weight.device)
    kv_caches.append(cache)

  packed_prompts, positions, cum_len, _ = pack_prompts(prompts, model.lm_head.weight.device, max_seq_len)
  init_last_positions = cum_len[1:] - 1
  n_prompt = len(filtered_prompts)
  prompt_not_end = torch.ones_like(
    init_last_positions, device=init_last_positions.device, dtype=torch.bool
  )

  generations = [[] for _ in range(len(prompts))]
  
  with torch.inference_mode():
    for i in range(max_new_tokens):
        if i == 0:
          embeddings = model(packed_prompts, positions=positions, caches=kv_caches, cum_seq_lens=cum_len, max_prompt_len=max_seq_len)
          last_embedding = embeddings[init_last_positions, :]
        else:
          q_cum_len = torch.arange(packed_prompt_len.size()[0] + 1, dtype=torch.long, device=model.lm_head.weight.device)
          embeddings = model(packed_prompts, positions=positions, caches=kv_caches, cum_seq_lens=q_cum_len, max_prompt_len=max_seq_len)
          last_embedding = embeddings

        logits = model.compute_logits(last_embedding)
        packed_prompts = torch.argmax(logits, dim=-1)

        update_idx_lst = is_kept_idx[prompt_not_end].tolist()
        packed_prompts_lst = packed_prompts.list()
        for idx, token_id in zip(update_idx_lst, packed_prompts_lst):
          generations[idx].append(token_id)

        generated_prompt_len = torch.where(
          prompt_not_end, 1, 0
        )
        cum_len[1:] += generated_prompt_len.cumsum(dim=-1)
        packed_prompt_len = (cum_len[1:] - cum_len[:-1])[prompt_not_end]

        prompt_not_end[prompt_not_end] = (packed_prompts != eos_token_id) & (cum_len[1:] - cum_len[:1] < max_seq_len)

        positions = init_last_positions + (i + 1)
        positions = positions[prompt_not_end]

        packed_prompts = packed_prompts[(packed_prompts != eos_token_id) & (packed_prompt_len < max_seq_len)]


  return generations
        

      


      



