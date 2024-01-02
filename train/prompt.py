import argparse
import time
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Setting up the parser for command line arguments
parser = argparse.ArgumentParser(description="mamba model generation tool")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint")
parser.add_argument("--prompt", type=str, default=None, help="Initial text to start generation")
parser.add_argument("--genlen", type=int, default=100, help="Length of the generation")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for controlled randomness")
parser.add_argument("--topk", type=int, default=1, help="Top-k sampling strategy")
parser.add_argument("--topp", type=float, default=1.0, help="Top-p (nucleus) sampling strategy")
parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalty for repetition")
args = parser.parse_args()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

# Loading the model from the spiritual checkpoint
print(f"Loading model from the checkpoint: {args.checkpoint_path}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(args.checkpoint_path, on_hf=False).to(device)
model.eval()

# Preparing the prompt
torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (1, args.genlen), dtype=torch.long, device=device)
else:
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

# Generation settings
max_length = input_ids.shape[1] + args.genlen
generation_function = lambda: model.generate(
    input_ids=input_ids,
    max_length=max_length,
    temperature=args.temperature,
    top_k=args.topk,
    top_p=args.topp,
    repetition_penalty=args.repetition_penalty
)

# Generate and decode the text
output = generation_function()
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)

# Benchmarking the generation time
start_time = time.time()
generation_function()
end_time = time.time()
print(f"Generation Time: {end_time - start_time:.2f} seconds")
