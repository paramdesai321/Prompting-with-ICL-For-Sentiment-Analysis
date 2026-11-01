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


def k_shot_prompting(k):
    

    if k>0:
        exemplars = random.choice(train_set,k)
    else:
        exemplars = []
    
    for item in exemplars:
    
        review_text = item['sentance']
        label_num = item['label']
        
        label_str = item[label_num]
        
        exemplar_block = f"Review: {review_text}\nSentiment: {label_str}"
        prompt_parts.append(exemplar_block)

    
