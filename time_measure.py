import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm import Mamba
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


device = "cuda:5"
x = 'hello world'
x2 = 'hello world2'
chunk = 'hello world ' * 300
question = 'What is the meaning of life?'
dtype=torch.bfloat16
repeats = 5
x3 = torch.rand(2, 64, 16, device=device)
torch.random.manual_seed(0)
model_path = 'state-spaces/mamba-2.8b'
model = MambaLMHeadModel.from_pretrained(model_path, device=device, dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", device=device, dtype=dtype)
model.requires_grad_(False)
model.eval()
input = tokenizer(chunk, truncation=False, return_tensors="pt").to(device)
input2 = tokenizer(question, truncation=False, return_tensors="pt").to(device)
input_ids2 = torch.randint(high=32000, size=(20, 20), dtype=torch.long, device=device)

times = 0
max_length = input2['input_ids'].shape[1] + 100
with torch.inference_mode():

    for repeat in range(repeats):
        #x = torch.randint(high=32000, size=(1, 10000), device=device)
        inference_all = InferenceParams(max_seqlen=5000, max_batch_size=20)
        inference_all.seqlen_offset = input['input_ids'].shape[1]
        # y_all = model.generate(input_ids=input2['input_ids'], max_length=max_length, cg=True,
        #                        return_dict_in_generate=True, output_scores=True)
        y_all = model(input_ids=input['input_ids'])
        t1 = time.time()
        for i in range(input_ids2.shape[1]): #for i in range(input2['input_ids'].shape[1]):
            inference_all.seqlen_offset = input_ids2.shape[1] + i + 1 #input['input_ids'].shape[1] + i + 1
            y_with_inference = model(input_ids=input_ids2[:, i:i+1], inference_params=inference_all) #model(input_ids=input2['input_ids'][:, i:i+1], inference_params=inference_all)
        t2 = time.time()
        inference_time = t2 - t1
        times += inference_time
        print(f'{model_path}, Forward: inference time: {inference_time}')
    times /= repeats
    print(f"1: Average time is : {times}")
    
    times = 0
    for repeat in range(repeats):
        #x = torch.randint(high=32000, size=(1, 10000), device=device)
        inference_all = InferenceParams(max_seqlen=5000, max_batch_size=20)
        inference_all.seqlen_offset = input['input_ids'].shape[1]
        y_all = model(input_ids=input_ids2)
        t1 = time.time()
        y_with_inference = model(input_ids=input_ids2, inference_params=inference_all) #model(input_ids=input2['input_ids'], inference_params=inference_all)
        t2 = time.time()
        inference_time = t2 - t1
        times += inference_time
        print(f'{model_path}, Forward: inference time: {inference_time}')
    times /= repeats
    print(f"2: Average time is : {times}")
