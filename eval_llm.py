import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from datasets import load_dataset
from evaluate import load
import pandas as pd

bertscore = load("bertscore")
bleu = load("bleu")
rouge = load('rouge')

def transform(example):
    instruction = example['instruction']
    input = example['input']
    example['prompt'] = f'{instruction}\n{input}'
    return example
    
dataset = load_dataset("llm-blender/mix-instruct", split = 'test')
dataset = dataset.map(transform)


model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
)

sequences = pipeline(
    dataset['prompt'],
    max_length=256,
    do_sample = True,
    top_p=0.7,
    eos_token_id=tokenizer.eos_token_id,
)

gen_text = [sequences[i][0]['generated_text'].replace(dataset['prompt'][i] + '\n', '', 1) for i in range(len(dataset))]
bertscore_results = bertscore.compute(predictions=gen_text, references=dataset['output'], lang="en")
bleu_results = [bleu.compute(predictions = gen_text[i], references = dataset['output'][i]) for i in range(len(gen_text))]
rouge_results = rouge.compute(predictions = gen_text, references = dataset['output'])

data = pd.DataFrame(columns=['Prompt', 'Result', 'Output', 'BLEU Score', 'Rouge Score'])

with open('falcon-7b-eval.json', 'w') as f:
    json.dump(json_dict, f, indent=4)