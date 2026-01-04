import json
import random
import string
import os

def get_wordbug_typo(text, prob):
    words = text.split()
    perturbed_words = []
    for word in words:
        if len(word) >= 3 and random.random() < prob:
            bug_type = random.choice(['swap', 'substitute', 'delete', 'insert'])
            word_list = list(word)
            if bug_type == 'swap':
                idx = random.randint(0, len(word_list) - 2)
                word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
            elif bug_type == 'substitute':
                idx = random.randint(0, len(word_list) - 1)
                word_list[idx] = random.choice(string.ascii_lowercase)
            elif bug_type == 'delete':
                idx = random.randint(0, len(word_list) - 1)
                word_list.pop(idx)
            elif bug_type == 'insert':
                idx = random.randint(0, len(word_list))
                word_list.insert(idx, random.choice(string.ascii_lowercase))
            perturbed_words.append("".join(word_list))
        else:
            perturbed_words.append(word)
    return " ".join(perturbed_words)

def get_intensive_leet(text, prob):
    leet_map = {
        'a': ['4', '@', '/-\\'], 'b': ['8', '|3'], 'c': ['(', '<', '{'], 'd': ['|)', '[)', '|>'],
        'e': ['3', '&'], 'g': ['6', '9'], 'h': ['#', '|-|', '[-]'], 'i': ['1', '!', '|'],
        'l': ['1', '|_', '7'], 'm': ['|\\/|', '/\\/\\', '(V)'], 'n': ['|\\/', '/\\/'], 'o': ['0', '()', '[]'],
        'p': ['|D', '|*', '|o'], 'r': ['|2', '2', '[2'], 's': ['5', '$', 'z'], 't': ['7', '+', "']['"],
        'u': ['|_|', '(_)'], 'v': ['\\/', '|/'], 'w': ['\\/\\/', '\\X/', 'vv'], 'x': ['><', '}{'], 'z': ['2', '7_']
    }
    res = []
    for char in text:
        low_char = char.lower()
        if low_char in leet_map and random.random() < prob:
            res.append(random.choice(leet_map[low_char]))
        else:
            res.append(char)
    return "".join(res)

def get_intensive_unicode(text, prob):
    unicode_map = {
        'a': ['а', 'ɑ', 'α'], 'b': ['Ь', 'в', 'ʙ'], 'c': ['с', 'ⅽ', 'ϲ'], 'd': ['ԁ', 'ⅾ'],
        'e': ['е', 'ϵ'], 'g': ['ɡ', 'ց'], 'h': ['һ', 'н', 'ɦ'], 'i': ['і', 'ɩ', 'ⅰ'],
        'k': ['к', 'κ'], 'l': ['ӏ', 'ⅼ'], 'm': ['ⅿ', 'м'], 'n': ['ո', 'п'],
        'o': ['о', 'ο', 'σ'], 'p': ['р'], 'r': ['г'], 's': ['ѕ'], 't': ['т', 'τ'],
        'u': ['υ', 'ս'], 'v': ['ν', 'ѵ', 'ⅴ'], 'x': ['х', 'ⅹ'], 'y': ['у', 'γ']
    }
    res = []
    for char in text:
        low_char = char.lower()
        if low_char in unicode_map and random.random() < prob:
            rep = random.choice(unicode_map[low_char])
            res.append(rep.upper() if char.isupper() else rep)
        else:
            res.append(char)
    return "".join(res)

def get_whitespace(text, prob):
    result = []
    for char in text:
        result.append(char)
        if char != ' ' and random.random() < prob:
            result.append(' ')
    return "".join(result)

def process_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    final_output = []
    intensities = [0.1, 0.20, 0.30]
    labels = ["low", "medium", "high"]

    for entry in dataset:
        final_output.append({
            "prompt": entry["prompt"],
            "type": entry.get("type"),
            "classification": entry.get("classification"),
            "pertubation_type": "none"
        })

        for i in range(3):
            p = intensities[i]
            text = entry["prompt"]
            
            text = get_wordbug_typo(text, p)
            text = get_intensive_leet(text, p*0.5)
            text = get_intensive_unicode(text, p*0.5)
            text = get_whitespace(text, p * 0.3)

            final_output.append({
                "prompt": text,
                "type": entry.get("type"),
                "classification": entry.get("classification"),
                "pertubation_type": labels[i]
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

input_file = r"LLM\PROJECT\PromptScreen\offence\dedup.json"
output_file = r"LLM\PROJECT\PromptScreen\svm_ablation_study\perturbed_output.json"
process_dataset(input_file, output_file)