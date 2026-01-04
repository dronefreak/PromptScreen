import json
import random

def split_dataset(input_file, test_file, train_file, test_ratio=0.06):
    # 1. Load the dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # 2. Shuffle the data to ensure randomness
    random.seed(42) # Optional: Set seed for reproducibility
    random.shuffle(data)

    # 3. Calculate the split point
    total_count = len(data)
    test_count = int(total_count * test_ratio)
    
    # 4. Split the list
    test_data = data[:test_count]
    train_data = data[test_count:]

    # 5. Save the test file
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    # 6. Save the train file
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)

    print(f"Successfully processed {total_count} items.")
    print(f"Saved {len(test_data)} items to {test_file} (6%)")
    print(f"Saved {len(train_data)} items to {train_file} (94%)")

# Execute the function
split_dataset(r"C:\Users\tvijo\Desktop\coding\LLM\PROJECT\PromptScreen\svm_ablation_study\perturbed_dedup.json", r"C:\Users\tvijo\Desktop\coding\LLM\PROJECT\PromptScreen\svm_ablation_study\metrics_test.json",r"C:\Users\tvijo\Desktop\coding\LLM\PROJECT\PromptScreen\svm_ablation_study\metrics_train.json")