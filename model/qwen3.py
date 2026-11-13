import torch
import model.layers as layers
import torch.nn as nn
from model.cache import KVCache

class Qwen3Model(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      num_hidden_layers: int,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float,
      rope_theta: float,
      max_seq_len: int
  ):
    super().__init__()
    self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    self.layers = nn.ModuleList([
      layers.Qwen3DecoderLayer(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_seq_len,
      ) for _ in range(num_hidden_layers)
    ])

    self.norm = layers.RMSNorm(hidden_size, rms_norm_eps)


  def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, caches: list[KVCache] | None=None, cum_len: torch.Tensor | None=None, max_seq_len: int | None=None) -> torch.Tensor:
    embedding = self.embed_tokens(input_ids)
    
    if caches is None:
      caches = [None for _ in self.layers]
    
    for layer, cache in zip(self.layers, caches):
      embedding = layer(embedding, positions, cache=cache, cum_len=cum_len, max_seq_len=max_seq_len)

    return self.norm(embedding)


class Qwen3ForCausalLM(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      num_hidden_layers: int,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float,
      rope_theta: float,
      max_seq_len: int,
      tie_word_embeddings: bool
  ):
    super().__init__()
    self.num_hidden_layers = num_hidden_layers
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.model = Qwen3Model(
      vocab_size,
      num_hidden_layers,
      hidden_size,
      intermediate_size,
      num_heads,
      num_kv_heads,
      head_dim,
      rms_norm_eps,
      rope_theta,
      max_seq_len
    )

    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    if tie_word_embeddings:
      self.lm_head.weight = self.model.embed_tokens.weight
    
  def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, caches: list[KVCache] | None=None, cum_seq_lens: torch.Tensor | None=None, max_prompt_len: int | None=None) -> torch.Tensor:
    return self.model(input_ids, positions, caches=caches, cum_len=cum_seq_lens, max_seq_len=max_prompt_len)

  def compute_logits(self, hidden_states: torch.Tensor):
    return self.lm_head(hidden_states)

    
  
    
    
