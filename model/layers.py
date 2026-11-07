import torch
from torch import nn
from torch.distributed import get_rank, get_world_size
from flash_attn import flash_attn_func
from model.cache import KVCache

class LinearBase(nn.Module):
  def __init__(
      self,
      input_size: int,
      output_size: int,
      bias: bool = False,
      tp_dim: int | None = None
  ):
    super().__init__()
    self.tp_dim = tp_dim
    self.weight = torch.nn.Parameter(
      torch.empty(output_size, input_size)
    )
    self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    self.tp_rank = get_rank()
    self.tp_size = get_world_size()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class ReplicatedLinear(LinearBase):
  def __init__(
      self,
      input_size: int,
      output_size: int,
      bias: bool = False
  ):
    super().__init__(input_size, output_size, bias)

  def weight_loader(self, param: nn.Parameter, load_weight: torch.Tensor) -> None:
    if param.size() != load_weight.size():
      raise ValueError(f"source and destination shape should be the same! Source shape: {load_weight.size()}, Destination shap:E {param.size()}")
    param.data.copy_(load_weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
  def __init__(
      self,
      input_size: int,
      output_size: int,
      bias: bool = False
  ):
    tp_size = get_world_size()
    if output_size % tp_size != 0:
      raise ValueError(f"output size should be divisible by world size. output size: {output_size}, world size: {tp_size}")
    super().__init__(input_size, output_size // tp_size, bias)

  def check_sizes(self, size1: torch.Size, size2: torch.Size):
   if len(size1) == 0:
      raise ValueError("size1 has zero dimension!")
   
   if len(size2) == 0:
      raise ValueError("size2 has zero dimension!")
   
   if len(size1) != len(size2):
    raise ValueError(f"size1 and size2 has different number of dimensions. size1 dimensions: {len(size1)}, size2 dimensions: {len(size2)}")

  def weight_loader(self, param: nn.Parameter, load_weight: torch.Tensor) -> None:
    param_size = param.size()
    weight_size = load_weight.size()
    
    self.check_sizes(param_size, weight_size)

    if param_size[0] * self.tp_size != weight_size[0]:
      raise ValueError(f"Cannot shard source tensor: its size along the partition dimension must equal the destination tensor’s size multiplied by the world size.")

    param.data.copy_(load_weight[self.tp_rank * param_size[0] : (self.tp_rank + 1) * param_size[0]])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
  def __init__(
      self,
      input_size: int,
      output_sizes: list[int],
      bias: bool = False
  ):
    super().__init__(input_size, sum(output_sizes), bias)
    for s in output_sizes:
      if s % self.tp_size != 0:
        raise ValueError("output size can't be divided by world size!")

    self.chunk_boundaries = [0]
    for s in output_sizes:
      self.chunk_boundaries.append(self.chunk_boundaries[-1] + s // self.tp_size)

  def weight_loader(self, param: nn.Parameter, load_weight: torch.Tensor, shard_id: int) -> None:
    param_size = param.size()
    weight_size = load_weight.size()

    self.check_sizes(param_size, weight_size)

    if weight_size[0] % self.tp_size != 0:
      raise ValueError("weight parameter size can't be divided by world size!")
    
    chunk_size = weight_size[0] // self.tp_size
    weight_chunk = load_weight[self.tp_rank * chunk_size: (self.tp_rank + 1) * chunk_size]

    param_chunk_size = self.chunk_boundaries[shard_id + 1] - self.chunk_boundaries[shard_id]
    if chunk_size != param_chunk_size:
          raise ValueError("chunk size doesn't match! Copy can't be done!")
    
    param_chunk = param.data[self.chunk_boundaries[shard_id] : self.chunk_boundaries[shard_id + 1]]
    param_chunk.copy_(weight_chunk)


class RMSNorm(nn.Module):
  def __init__(
      self,
      hidden_size: int,
      eps: float
  ):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(
      torch.ones(hidden_size)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    return x / norm * self.weight


class SwiGLU(nn.Module):
  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
  ):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = self.gate_proj(x)
    intermediate = gate * torch.sigmoid(gate) * self.up_proj(x)
    return self.down_proj(intermediate)

  
class RotaryEmbedding(nn.Module):
  def __init__(
      self,
      head_dim: int,
      theta: float,
      max_seq_len: int
  ):
    assert head_dim > 0, "head_dim must be positive"
    assert head_dim % 2 == 0, "head_dim must be even"
    assert theta > 0, "theta must be postive"
    assert max_seq_len > 0, "maximum sequence length should be positive integer"
    super().__init__()

    self.head_dim = head_dim
    self.theta = theta
    theta_exponential = self.theta ** (-2 * torch.arange(self.head_dim // 2) / self.head_dim)
    angle = torch.arange(max_seq_len).view(-1, 1) * theta_exponential.view(1, -1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    rotation = torch.stack([cos, -sin, sin, cos], dim=-1)
    self.register_buffer("rotation", rotation.view(max_seq_len, head_dim // 2, 2, 2))
    
  def forward(self, x : torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
    """
    Args:
      x: Floating tensor [batch, sequence, heads, head_dim]
      positions: Integer tensor [batch, sequence]

    Returns:
      Embedding after RoPE
    """
    x_size = x.size()
    position_size = positions.size()

    if position_size[0] != x_size[0]:
      raise ValueError("Batch size don't match!")

    if position_size[1] != x_size[1]:
      raise ValueError("Sequence length don't match!")

    if x_size[-1] != self.head_dim:
      raise ValueError(f"require the head_dim of the input tensor to be {self.head_dim}")
    x_reshape = x.view(*x_size[:-1], 2, x_size[-1] // 2).transpose(-2, -1)
    x_reshape = x_reshape.view(*x_reshape.size()[:-1], 2, 1)

    rotation = self.rotation[positions, :, :, :]
    rotation = rotation.view(x_size[0], x_size[1], 1, self.head_dim // 2, 2, 2)
    result = torch.matmul(rotation, x_reshape)
    result = result.view(result.size()[:-1]).transpose(-2, -1).reshape(*result.size()[:-2], -1)

    return result.view(*result.size()[:-2], -1)


class Qwen3Attention(nn.Module):
  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float,
      rope_theta: float,
      max_seq_len: int
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.num_kv_heads = num_kv_heads
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    self.q_norm = RMSNorm(head_dim, rms_norm_eps)
    self.k_norm = RMSNorm(head_dim, rms_norm_eps)
    self.rotary_emb = RotaryEmbedding(head_dim, rope_theta, max_seq_len)

  def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: KVCache | None=None) -> torch.Tensor:
    q = self.q_proj(x)
    q = q.view(*q.size()[:-1], self.num_heads, self.head_dim)
    k = self.k_proj(x)
    k = k.view(*k.size()[:-1], self.num_kv_heads, self.head_dim)
    v = self.v_proj(x)
    v = v.view(*v.size()[:-1], self.num_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)

    q = self.rotary_emb(q, positions)
    k = self.rotary_emb(k, positions)

    if cache:
      if cache.length == 0:
        cache.to(k.dtype)
      k, v = cache.append(k, v)
      
    attn = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    attn = attn.view(*x.size()[:-1], -1)

    return self.o_proj(attn)

    
class Qwen3DecoderLayer(nn.Module):
  def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    rope_theta: float,
    max_seq_len: int,
  ):
    super().__init__()
    self.self_attn = Qwen3Attention(
      hidden_size,
      num_heads,
      num_kv_heads,
      head_dim,
      rms_norm_eps,
      rope_theta,
      max_seq_len
    )

    self.mlp = SwiGLU(
      hidden_size,
      intermediate_size
    )

    self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)


  def forward(self, x:torch.Tensor, positions: torch.Tensor, cache: KVCache | None=None) -> torch.Tensor:
    update = self.self_attn(self.input_layernorm(x), positions, cache=cache)
    x_update = x + update
    return x_update + self.mlp(self.post_attention_layernorm(x_update))





    

    