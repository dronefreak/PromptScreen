import os, json
from datasets import load_dataset

DATASET_NAME = "TrustAIRLab/in-the-wild-jailbreak-prompts"
OUTPUT_FILE = "data/threats.json" # We'll save the formatted data here

def main():
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face Hub...")

    try:
        dataset = load_dataset(DATASET_NAME, "jailbreak_2023_05_07", split="train")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    threats_for_db = []
    print("Processing prompts...")

    for entry in dataset:
        prompt_text = entry.get("prompt")
        source_category = entry.get("source", "unknown")
        if prompt_text:
            threats_for_db.append({
                "prompt": prompt_text,
                "category": source_category
            })

    print(f"Processed {len(threats_for_db)} prompts.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(threats_for_db, f, indent=2)

    print(f"Successfully saved formatted threat data to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
