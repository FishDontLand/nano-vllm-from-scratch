import torch
import torch.nn as nn
import numpy as np
from model import batching
import timeit
import generation
import argparse

from transformers import AutoTokenizer
from utils import loader
from pathlib import Path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_test_config(path: str, batch_choice: list[list[int]]) -> tuple[nn.Module, AutoTokenizer, list[list[list[int]]]]:
    np.random.seed(23123)

    path = Path(path)
    config_path = path / 'config.json'
    model = loader.build_model_from_config(config_path)
    loader.load_weights(model, path)
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)

    token_ids = []
    for batch_seq in batch_choice:
        prompts = []
        for seq_len in batch_seq:
            sample = np.random.randint(
                low=0,
                high=len(tokenizer) - 1,
                size=seq_len,
            )
            
            sample[sample >= tokenizer.eos_token_id] += 1
            sample = sample.tolist()
            prompts.append(sample)
        token_ids.append(prompts)

    return model, tokenizer, token_ids
    

def benchmark(model: nn.Module, tokenizer: AutoTokenizer, 
              batch_prompts: list[list[list[int]]], return_last_embedding: bool=False, method_types: list[str]=['batch', 'pack', 'oneByOne']):

    warmup_steps = 5
    eval_steps = 10

    benchmark_stat = {
        m : {
            'prefill': dict(),
            'throughput': dict(),
            'ttft': dict(),
        } for m in method_types
    }

    last_embedding_cache = {
        m : [None for _ in batch_prompts] for m in method_types
    }

    warm_up_done = False
    with torch.inference_mode():
        for method in method_types:
            for prompt_id, prompts in enumerate(batch_prompts):
                num_tokens = sum([len(p) for p in prompts])
                if method == 'batch':
                    prompts, prompt_len = batching.batch_prompts(prompts, device=DEVICE)
                elif method == "pack":
                    prompts, positions, cum_len, longest_prompt_len = batching.pack_prompts(prompts, device=DEVICE)
                elif method != "oneByOne":
                    raise ValueError(f"Unrecognized method: {method}")

                if not warm_up_done and method != 'oneByOne':
                    for _ in range(warmup_steps):
                        if method == "batch":
                            model(prompts)
                        else:
                            model(prompts, positions, cum_seq_lens=cum_len, max_prompt_len=longest_prompt_len)

                    warm_up_done = True
                
                torch.cuda.synchronize()


                times = []
                for _ in range(eval_steps):
                    start = timeit.default_timer()
                    if method == "batch":
                        embeddings = model(prompts)
                    elif method == "pack":
                        embeddings = model(prompts, positions, cum_seq_lens=cum_len, max_prompt_len=longest_prompt_len)
                    elif method == "oneByOne":
                        break
                    else:
                        raise ValueError(f"Unrecognized Method: {method}")

                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    times.append(end - start)

                prefill_time = sum(times) / eval_steps

                if method != 'oneByOne':
                    benchmark_stat[method]['prefill'][prompt_id]  = prefill_time
                    benchmark_stat[method]['throughput'][prompt_id] = num_tokens / prefill_time


                times = []
                
                for _ in range(eval_steps):
                    start = timeit.default_timer()
                    if method == "batch":
                        embeddings = model(prompts)
                        last_embeddings = torch.gather(embeddings, dim=1, 
                                                    index=torch.tensor(
                                                        [[[p - 1] * embeddings.size()[-1]] for p in prompt_len],
                                                        dtype=torch.long,
                                                        device=DEVICE
                                                    ))
                        last_embeddings = last_embeddings.view(len(prompts), last_embeddings.size()[-1])

                    elif method == "pack":
                        embeddings = model(prompts, positions, cum_seq_lens=cum_len, max_prompt_len=longest_prompt_len)
                        last_loc = [l - 1 for l in cum_len[1:]]
                        last_embeddings = embeddings[last_loc, :]

                    elif method == 'oneByOne':
                        last_embeddings = []
                        for p in prompts:
                            _, embedding = generation.generate_greedy(
                                model,
                                torch.tensor([p], dtype=torch.long, device=DEVICE),
                                1,
                                tokenizer.eos_token_id,
                                max_seq_len=1024,
                                use_cache=False,
                                return_embedding=True
                            )
                            last_embeddings.append(embedding[0])

                        last_embeddings = [e.view(1, -1) for e in last_embeddings]
                        last_embeddings = torch.concat(last_embeddings, dim=0)
                            
                    else:
                        raise ValueError(f"Unrecognized method: {method}")

                    if last_embedding_cache[method][prompt_id] is None and return_last_embedding:
                        last_embedding_cache[method][prompt_id] = last_embeddings

                    if method == 'oneByOne':
                        break

                    logits = model.compute_logits(last_embeddings)
                    max_arg = torch.argmax(logits, dim=-1)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    times.append(end - start)

                ttft = sum(times) / eval_steps
                if method != 'oneByOne':
                    benchmark_stat[method]['ttft'][prompt_id] = ttft

    return benchmark_stat, last_embedding_cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments for benchmarking")
    parser.add_argument("--path",
                        type=str,
                        required=True)
    
    parser.add_argument("--batch_size", 
                        help="sequence length for each batch",
                        action='append', nargs='+',
                        type=int, 
                        required=True)

    args = parser.parse_args()
    model, tokenizer, test_token_ids = load_test_config(args.path, args.batch_size)
    test_results, _ = benchmark(model, tokenizer, test_token_ids, method_types=['batch', 'pack'])
    print(test_results)









        
        