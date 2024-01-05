import torch
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


model = MambaLMHeadModel.from_pretrained('/data/norman_mu/models/mamba-1.4b', use_fast_path=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

pad_count = 10

# Check prefill logits
input_ids = torch.randint(1, 1000, (1, 1024)).to('cuda')
input_ids_padded = torch.cat([torch.zeros_like(input_ids[:, [0] * pad_count]), input_ids], dim=1)
attention_mask = torch.cat([torch.zeros_like(input_ids[:, [0] * pad_count]), torch.ones_like(input_ids)], dim=1)

out = model(input_ids_padded).logits.detach().cpu()
out_padded = model(input_ids_padded, attention_mask).logits.detach().cpu()
out_true = model(input_ids).logits.detach().cpu()

print("max L2 error:", (out_true - out[:, pad_count:]).norm(dim=-1).max())
print("max L2 errors (padded):", (out_true - out_padded[:, pad_count:]).norm(dim=-1).max())


# Check decoding outputs
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'

print("\n\nNo CUDA graph:")
inputs = tokenizer([text], return_tensors='pt').to('cuda')
x = model.generate(inputs.input_ids, max_length=100, temperature=0, cg=False)
print("\nNo pad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to('cuda')
x = model.generate(inputs.input_ids, max_length=100 + pad_count, temperature=0, cg=False)
print("\nPad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to('cuda')
inputs.attention_mask[:, :pad_count] = 0
x = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100 + pad_count, temperature=0, cg=False)
print("\nPad, mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

print("\n\nCUDA graph:")
inputs = tokenizer([text], return_tensors='pt').to('cuda')
x = model.generate(inputs.input_ids, max_length=100, temperature=0, cg=True)
print("\nNo pad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to('cuda')
x = model.generate(inputs.input_ids, max_length=100 + pad_count, temperature=0, cg=True)
print("\nPad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to('cuda')
inputs.attention_mask[:, :pad_count] = 0
x = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100 + pad_count, temperature=0, cg=True)
print("\nPad, mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))
