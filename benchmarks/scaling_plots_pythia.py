# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import os

if not os.path.exists("scaling_results"):
    os.mkdir("scaling_results")


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
# parser.add_argument("--promptlen", type=int, default=100)
# parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
# parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

model_name_str = args.model_name
model_name_str = model_name_str.replace("state-spaces/", "").replace("-", "_").replace("/", "_")

# TODO set back to 3
repeats = 3
device = "cuda"
# TODO fix fp16 support
dtype = torch.float32

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    print(model)
elif "pythia" in args.model_name:
    #tested with EleutherAI/pythia-70m-deduped
    # and EleutherAI/pythia-1.4b
    model = GPTNeoXForCausalLM.from_pretrained(
            args.model_name,
            # cache_dir=f"./{model_name_str}/cache",
            #cache_dir="./pythia-1.4b",
            device_map={"": device}, torch_dtype=dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            # cache_dir=f"./{model_name_str}/cache",
            # #cache_dir="./pythia-1.4b",
    )

else:
    raise ValueError(f"Untested/unsupported model {args.model_name}")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
# if args.prompt is None:
#     input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
#     attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
# else:
#     tokens = tokenizer(args.prompt, return_tensors="pt")
#     input_ids = tokens.input_ids.to(device=device)
#     attn_mask = tokens.attention_mask.to(device=device)
def get_time(promptlen, genlen, batch):
    input_ids = torch.randint(1, 1000, (batch, promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

    max_length = input_ids.shape[1] + genlen

    if is_mamba:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
    # warmup for gpu and memory allocation
    out = fn()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        out = fn()
    torch.cuda.synchronize()
    # print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
    # print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

    # if args.prompt is not None:
    #     print(tokenizer.batch_decode(out.sequences.tolist()))
    return (time.time() - start) / repeats * 1000

import pandas as pd

# # expt 1: t vs bs with const genlen(128) and promptlen(2048)
# print(f"running expt 1: t vs bs with const genlen(128) and promptlen(2048) for {model_name_str}")
# promptlen = 2048
# genlen = 128
# data = []
# for bs in [2**i for i in range(8)]:
#     t_measured = get_time(promptlen, genlen, bs)
#     data.append({"bs": bs, "time": t_measured})

#     # Convert the list of dictionaries to a DataFrame
#     df = pd.DataFrame(data)
#     # Save the DataFrame as a CSV file
#     df.to_csv(f"scaling_results/experiment_t_bs_{model_name_str}.csv", index=False)

# print("Finished experiment 1")

# expt 2: t vs promptlen with const bs(1) and genlen(1)
# print(f"running expt 2: expt 2: t vs promptlen with const bs(1) and genlen(1) for {model_name_str}")
# bs = 1
# genlen = 1
# data = []
# for promptlen in [2**i for i in range(18)]:
#     t_measured = get_time(promptlen, genlen, bs)
#     data.append({"promptlen": promptlen, "time": t_measured})

#     # Convert the list of dictionaries to a DataFrame
#     df = pd.DataFrame(data)
#     # Save the DataFrame as a CSV file
#     df.to_csv(f"scaling_results/experiment_t_promptlen_{model_name_str}.csv", index=False)

# print("Finished experiment 1")


# expt 3: t vs bs, promptlen with const genlen(1)
# print(f"running expt 3: t vs bs, promptlen with const genlen(1) for {model_name_str}")
# genlen = 1
# data = []
# for bs in [2**i for i in range(6)]:
#     for promptlen in [2**i for i in range(1, 8)]:
#         t_measured = get_time(promptlen, genlen, bs)
#         data.append({"promptlen": promptlen, "bs":bs, "time": t_measured})

#         # Convert the list of dictionaries to a DataFrame
#         df = pd.DataFrame(data)
#         # Save the DataFrame as a CSV file
#         df.to_csv(f"scaling_results/experiment_t_promptlen_bs_matrix_{model_name_str}.csv", index=False)
# print("Finished experiment 1")


# expt 4: t vs promptlen with const bs(1) and genlen(1), increase promptlen indefinitely
print(f"expt 4: t vs promptlen with const bs(1) and genlen(1), increase promptlen indefinitely for {model_name_str}")
bs = 1
genlen = 1
data = []
promptlen = 512
while True:
    t_measured = get_time(promptlen, genlen, bs)
    data.append({"promptlen": promptlen, "time": t_measured})

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame as a CSV file
    df.to_csv(f"scaling_results/experiment_t_promptlen_whileTrue_{model_name_str}.csv", index=False)

    promptlen *= 2
# print("Finished experiment 1")
