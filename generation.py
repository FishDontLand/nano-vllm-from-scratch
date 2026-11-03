import torch
from model.qwen3 import Qwen3ForCausalLM
import torch.nn as nn

def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    max_seq_len: int,
) -> list[int]:
  if input_ids.size()[-1] >= max_seq_len:
    return input_ids.cpu().tolist()

  init_seq_len = input_ids.size()[-1]
  with torch.inference_mode():
    for i in range(max_new_tokens):
      positions = torch.arange(0, input_ids.size()[-1])
      dim_except_last = input_ids.numel() // input_ids.size()[-1]
      positions = positions.repeat(dim_except_last).view(*input_ids.size())
      embeddings = model(input_ids, positions=positions.to(input_ids.device))
      last_embedding = embeddings[:, [-1], :]
      logits = model.compute_logits(last_embedding)
      max_arg = torch.argmax(logits, dim=-1)
      input_ids = torch.concat([input_ids, max_arg], dim=-1)

      if eos_token_id == max_arg.item():
        output_ids = input_ids[:, init_seq_len:]
        return output_ids.cpu().tolist()[0]

      if input_ids.size()[-1] > max_seq_len:
        raise ValueError(f"Generated more tokens than capacity!")

    output_ids = input_ids[:, init_seq_len:]
    return output_ids.cpu().tolist()[0]



