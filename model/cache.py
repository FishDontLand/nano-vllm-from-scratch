import torch.nn as nn
import torch

class KVCache(nn.Module):
    def __init__(
            self,
            batch_size: int,
            capacity: int,
            num_kv_heads: int,
            head_dim: int
        ):
        super().__init__()
        for name in ['k_cache', 'v_cache']:
            self.register_buffer(
                name=name,
                tensor=torch.empty(
                    batch_size, capacity, num_kv_heads, head_dim
                ),
                persistent=False
            )
        self.length = 0


    def append(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert k.size() == v.size(), "k, v tensors should have the same shape"

        num_tokens = k.size()[1]
        self.k_cache[:, self.length:(self.length + num_tokens), :, :] = k
        self.v_cache[:, self.length:(self.length + num_tokens), :, :] = v
        self.length += num_tokens

        return self.k_cache[:, :self.length, :, :], self.v_cache[:, : self.length, :, :]

    def reset(self)->None:
        self.length = 0
    

    

