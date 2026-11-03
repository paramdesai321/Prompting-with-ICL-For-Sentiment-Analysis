import random
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(device) 


local_dir = './Qwen3-0.6B-Base'
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.float32, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(local_dir)


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

train_set = load_json('./DATA/SST2/train.json')
val_set = load_json('./DATA/SST2/val.json')

print("Train set number: ", len(train_set))
print("Validation set number: ", len(val_set))


def k_shot_prompting(query_text, train_data, k, label_map={0: "Negative", 1: "Positive"}):
    

    if k>0:
        exemplars = random.sample(train_set,k)
    else:
        exemplars = []
    prompt_parts = [] 
    for item in exemplars:
    
        review_text = item['sentence']
        label_num = item['label']
        
        label_str = label_map[label_num]
        
        exemplar_block = f"Review: {review_text}\nSentiment: {label_str}"
        prompt_parts.append(exemplar_block)

    
    query_block = f"Review: {query_text}\nSentiment:"
    prompt_parts.append(query_block)

    full_prompt = "\n\n".join(prompt_parts)

    return full_prompt
def parse_sentiment(raw_output_text, prompt_prefix="Sentiment:"):
    if prompt_prefix in raw_output_text:
        completion = raw_output_text.split(prompt_prefix)[-1]
    else: 
        completion = raw_output_text

    first_line = completion.split('\n')[0].strip().lower()

    if first_line.startswith("positive"):
        return 1
    if first_line.startswith("negative"):
        return 0

    if "positive" in first_line:
        return 1
    if "negative" in first_line:
        return 0

    return -1       
      

def kshot_robustness_prompt(query_text, train_data, k=3, format_type="baseline"):
    exemplars = random.sample(train_data, k) if k > 0 else []
    parts = []

    if format_type == "label_shift_words":
        label_map = {0: "Bad", 1: "Good"}
    elif format_type == "label_shift_nums":
        label_map = {0: "0", 1: "1"}
    else:
        label_map = {0: "Negative", 1: "Positive"}

    for item in exemplars:
        text = item['sentence']
        lbl = label_map[item['label']]
        
        if format_type == "format_shift":
            parts.append(f"Input: {text} | Label: {lbl}")
        else:
            parts.append(f"Review: {text}\nSentiment: {lbl}")

    if format_type == "format_shift":
        parts.append(f"Input: {query_text} | Label:")
    else:
        parts.append(f"Review: {query_text}\nSentiment:")

    return "\n\n".join(parts) 
def parse_robust_label(raw_text, format_type):
    first_line = raw_text.split('\n')[0].lower()
    
    if format_type == "label_shift_words":
        if "good" in first_line: return 1
        if "bad" in first_line: return 0
    elif format_type == "label_shift_nums":
        if "1" in first_line: return 1
        if "0" in first_line: return 0
    else:
        if "positive" in first_line: return 1
        if "negative" in first_line: return 0


experiments = ["format_shift", "label_shift_words", "label_shift_nums"]
robustness_results = {}

for exp in experiments:
    correct = 0
    for item in tqdm(val_set, desc=f"Testing {exp}"):
        prompt = kshot_robustness_prompt(item['sentence'], train_set, k=5, format_type=exp)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
        
        # Decode only generated tokens
        gen_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_robust_label(gen_text, exp)
        
        if pred == item['label']:
            correct += 1

    acc = (correct / len(val_set)) * 100
    robustness_results[exp] = acc
    print(f"Experiment '{exp}': {acc:.2f}%")

print("\nRobustness Comparison:", robustness_results)
#
#k_values = [0, 1, 3, 5, 8, 16]
#accuracy_results = {}
#
#for k in k_values:
#    correct = 0
#    total = 0
#    
#    print(f"\nEvaluating for K = {k}-shot...")
#
#    for item in val_set:
#        query_text = item['sentence']
#        ground_truth = item['label']
#
#        # 1. Build the dynamic prompt for this validation item
#        prompt = k_shot_prompting(query_text=query_text, train_data=train_set, k=k)
#
#        # 2. Tokenize the prompt and push input_ids to model device
#        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#        # 3. Generate text (disable gradient calculation for faster execution)
#        with torch.no_grad():
#            output_ids = model.generate(**inputs, max_new_tokens=10)
#
#        # 4. Decode generated token IDs back to a string
#        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
#
#        # 5. Parse predicted sentiment integer (0 or 1)
#        pred_label = parse_sentiment(raw_output_text=raw_output)
#
#        # 6. Update score count if prediction matches ground truth
#        if pred_label == ground_truth:
#            correct += 1
#        total += 1
#
#    accuracy = correct / total
#    accuracy_results[k] = accuracy
#    print(f"K = {k} | Accuracy: {accuracy * 100:.2f}%")
#
#print("\nFinal Results Summary:", accuracy_results) 
