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
        self.length = None

    def append(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            length: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert k.size() == v.size(), "k, v tensors should have the same shape"

        if length is None:
            length = torch.tensor([k.size()[1]], dtype=torch.long, device=k.device)
            if self.length is None:
                self.length = torch.tensor([0], dtype=torch.long, device=k.device)

            old_length = self.length.sum()
        else:
            if self.length is None:
                self.length = torch.zeros_like(length, dtype=torch.long, device=k.device)
            old_length = self.length.sum()

        new_length = old_length + length.sum()

        self.k_cache[:, old_length:new_length, :, :] = k
        self.v_cache[:, old_length:new_length, :, :] = v


        idx = torch.arange(new_length, dtype=torch.long, device=k.device)
        cum_old_len = self.length.cumsum(dim=-1).tolist()
        cum_len = length.cumsum(dim=-1).tolist()

        new_idx = []
        for i in range(len(cum_len)):
            if i == 0:
                new_idx.append(idx[:cum_old_len[i]])
                new_idx.append(idx[old_length:cum_len[i]])
            else:
                new_idx.append(idx[cum_old_len[i - 1] : cum_old_len[i]])
                new_idx.append(idx[(old_length + cum_len[i - 1]) : (old_length + cum_len[i])])
        new_idx = torch.concat(new_idx, dim=-1)
        
        self.k_cache[:, :new_length, :, :] = self.k_cache[:, new_idx, :, :]
        self.v_cache[:, :new_length, :, :] = self.v_cache[:, new_idx, :, :]

        self.length += length


        return self.k_cache[:, :new_length, :, :], self.v_cache[:, : new_length, :, :], self.length.cumsum(dim=-1)

    def reset(self)->None:
        self.length = None
    

    

