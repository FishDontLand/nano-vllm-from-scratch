import torch

def pack_prompts(
    prompts: list[list[int]],
    device: torch.device,
    max_seq_len: int,        
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if max_seq_len <= 0:
        raise ValueError("Non positve max_seq_len!")
    
    if prompts:
        input_ids = []
        positions = []
        cum_len = [0]
        longest_prompt_len = 0
        for p in prompts:
            if p:
                if len(p) > max_seq_len:
                    raise ValueError("prompt length greater than max capacity!")
                input_ids += p
                positions += list(range(len(p)))
                cum_len.append(len(p))
                longest_prompt_len = max(longest_prompt_len, len(p))
            else:
                raise ValueError("Empty prompt in the prompt list")
        input_ids = torch.tensor(input_ids, dtyp=torch.long, device=device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        cum_len = torch.tensor(cum_len, dtype=torch.int32, device=device)
        return input_ids, positions, cum_len, longest_prompt_len
    else:
        raise ValueError("Empty prompt list")    