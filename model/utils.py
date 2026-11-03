import torch
import math

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
  x_normalized = x - torch.max(x, dim=i, keepdim=True).values
  exponential = torch.exp(x_normalized)
  return exponential / exponential.sum(dim=i, keepdim=True)

def dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    query_positions: torch.Tensor, key_lengths: torch.Tensor
  ) -> torch.Tensor:
  """
  Args:
    q: float tensor, [batch, query_heads, query_length, head_dim]
    k: float tenosr, [batch, kv_heads, max_key_len, head_dim]
    v: float tensor, [batch, kv_heads, max_key_len, head_dim]
    query_positions: absolute positions, integer tensor, [batch, query_length]
    key_lengths: valid key count per request, integer tensor, [batch]
  Returns:
    float tensor, Embedding after attention
  """

  q_size = q.size()
  batch, query_heads, query_length, head_dim = q_size[0], q_size[1], q_size[2], q_size[3]
  k_size = k.size()
  key_batch, kv_heads, max_key_len, key_head_dim = k_size[0], k_size[1], k_size[2], k_size[3]

  assert batch == key_batch, "batch size for query and key must match!"
  assert kv_heads == v.size()[1], "number of heads for k, v should match!"
  assert query_heads % kv_heads == 0, "query heads must be multiple of kv heads"

  group_size = query_heads // kv_heads
  q_grouped = q.view(batch, kv_heads, group_size, query_length, head_dim)
  k = k.view(batch, kv_heads, 1, max_key_len, head_dim)
  k_T = torch.transpose(k, -2, -1)

  product = torch.matmul(q_grouped, k_T) / math.sqrt(head_dim)

  mask = torch.zeros([batch, query_length, max_key_len], device=q.device)
  key_idx = torch.arange(max_key_len)

  position_condition = key_idx <= query_positions.view(*query_positions.size(), 1)
  key_length_condition = key_idx < key_lengths.view(-1, 1)
  key_length_condition = key_length_condition.view(batch, 1, max_key_len)
  condition = position_condition & key_length_condition

  mask = torch.where(condition, mask, -torch.inf)
  weights = softmax(product + mask.view(batch, 1, 1, query_length, max_key_len), -1)
  attn_embedding = torch.matmul(weights, v.view(batch, kv_heads, 1, max_key_len, head_dim))

  return attn_embedding.view(batch, -1, query_length, head_dim)




  

