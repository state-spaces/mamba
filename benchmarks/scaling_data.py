import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--repeats", type=int, default=3, help="Number of times to repeat the generation for timing.")
parser.add_argument("--save-directory", type=str, default="./", help="Directory to save the output CSV files.")
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="Data type for model and inputs.")
parser.add_argument("--expt", type=int)

args = parser.parse_args()

repeats = args.repeats
device = "cuda"
dtype = torch.float32 if args.dtype == "float32" else torch.float16
dtype_str = str(dtype).replace("torch.", "")

def format_name(name):
    return name.lower().replace(' ', '_').replace('.', '_').replace('/', '_').replace('-', '_')

formatted_device_name = format_name(torch.cuda.get_device_name())
formatted_model_name = format_name(args.model_name)

# Constructing model_name_str with all elements
model_name_str = f"{formatted_model_name}__{formatted_device_name}__{dtype_str}"

print(model_name_str)

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
    fn()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats * 1000

import pandas as pd
import gc

# Example for using the save_directory
save_directory = args.save_directory
# Ensure the save_directory ends with a slash
if not save_directory.endswith("/"):
    save_directory += "/"

# Your existing experiment code goes here, just modify the saving part to use save_directory
# For example:
# df.to_csv(f"{save_directory}experiment_t_bs_{model_name_str}.csv", index=False)

if args.expt == 1:

    # expt 1: t vs bs with const genlen(128) and promptlen(2048)
    promptlen = 2048
    genlen = 128
    data = []
    for bs in [2**i for i in range(8)]:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            t_measured = get_time(promptlen, genlen, bs)
            data.append({"bs": bs, "time": t_measured})
            # incremental saving
            df = pd.DataFrame(data)
            df.to_csv(f"{save_directory}experiment_t_bs__{model_name_str}.csv", index=False)
        except Exception as e:
            print("try failed")
            print(e)
            print("breaking out of loop")
            break

elif args.expt == 2:
    # expt 2: t vs promptlen with const bs(1) and genlen(1)
    bs = 1
    genlen = 1
    data = []
    for promptlen in [2**i for i in range(18)]:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            t_measured = get_time(promptlen, genlen, bs)
            data.append({"promptlen": promptlen, "time": t_measured})
            df = pd.DataFrame(data)
            df.to_csv(f"{save_directory}experiment_t_promptlen__{model_name_str}.csv", index=False)
        except Exception as e:
            print("try failed")
            print(e)
            print("breaking out of loop")
            break

else:
    print(f"expt {args.expt} not found") 