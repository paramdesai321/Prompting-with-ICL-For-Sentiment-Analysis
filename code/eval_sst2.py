import random
import json

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

    
    query_block = f"Review: {query_text}\nSentiment:}"
    prompt_parts.append(query_block)

    full_prompt = "\n\n".join(prompt_parts)

    return full_prompt
    
