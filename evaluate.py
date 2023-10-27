from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from datasets import load_dataset

def transform(example):
    instruction = example['instruction']
    input = example['input']
    example['prompt'] = f'{instruction}\n{input}'
    return example
    

dataset = load_dataset("llm-blender/mix-instruct", split = 'test')
dataset = dataset.map(transform)


for example in dataset:
    print(example.keys())
    print(example['prompt'])
    break

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
)

result = []
for idx, example in enumerate(dataset):
    prompt = example['prompt']
    print(prompt)
    sequences = pipeline(
        prompt,
        max_length=70,
        do_sample = True,
        top_p=0.4,
        eos_token_id=tokenizer.eos_token_id,
    )
    result.append(sequences)
    if idx == 3:
        break

print(result[:3])