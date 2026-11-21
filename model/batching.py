import torch

def pack_prompts(
    prompts: list[list[int]],
    device: torch.device,
    max_seq_len: int | None=None,        
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if max_seq_len is None:
        max_seq_len = max([len(p) for p in prompts])
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
                cum_len.append(cum_len[-1] + len(p))
                longest_prompt_len = max(longest_prompt_len, len(p))
            else:
                raise ValueError("Empty prompt in the prompt list")
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        cum_len = torch.tensor(cum_len, dtype=torch.int32, device=device)
        return input_ids, positions, cum_len, longest_prompt_len
    else:
        raise ValueError("Empty prompt list")

def batch_prompts(
        prompts: list[list[int]],
        device: torch.device,
        max_seq_len: int | None=None
)-> tuple[torch.Tensor, list[int]]:
    n_prompts = len(prompts)
    if max_seq_len is None:
        max_seq_len = max([len(p) for p in prompts])

    processed_prompts = []
    prompt_len = []
    for p in prompts:
        processed_prompts.append(p + [0] * (max_seq_len - len(p)))
        prompt_len.append(len(p))
    
    batch = torch.tensor(processed_prompts, dtype=torch.long, device=device)

    return batch, prompt_len

    