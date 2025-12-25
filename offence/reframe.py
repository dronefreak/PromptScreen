import random

import pandas as pd


def perturb_prompt(prompt, num_perturbations=2, perturbation_rate=0.3):
    """Perturb words in prompt to simulate spelling mistakes"""
    words = prompt.split()
    perturbed_words = []

    for word in words:
        # Skip short words/numbers (they don't get typos)
        if len(word) <= 3 or word.isdigit():
            perturbed_words.append(word)
            continue

        # Decide if this word gets perturbed (30% chance)
        if random.random() > perturbation_rate:  # nosec
            perturbed_words.append(word)
            continue

        # Perturbation strategies (pick one randomly) # nosec
        perturbation_type = random.choice(
            ["swap", "delete", "insert", "replace"]
        )  # nosec

        if perturbation_type == "swap":  # teh → the
            if len(word) >= 2:
                i, j = random.sample(range(len(word)), 2)  # nosec B311
                chars = list(word)
                chars[i], chars[j] = chars[j], chars[i]
                perturbed = "".join(chars)

        elif perturbation_type == "delete":  # attractive → atrative
            idx = random.randint(0, len(word) - 2)  # nosec
            perturbed = word[:idx] + word[idx + 1 :]

        elif perturbation_type == "insert":  # prompt → propmpt
            idx = random.randint(1, len(word) - 1)  # nosec
            insert_char = random.choice("aeiou")  # nosec
            perturbed = word[:idx] + insert_char + word[idx:]

        elif perturbation_type == "replace":  # beautiful → beutiful
            vowels = "aeiou"
            idx = random.randint(1, len(word) - 2)  # nosec
            if word[idx] in vowels:
                perturbed = (
                    word[:idx]
                    + random.choice(vowels.replace(word[idx], ""))  # nosec
                    + word[idx + 1 :]
                )
            else:
                perturbed = (
                    word[:idx] + random.choice(vowels) + word[idx + 1 :]  # nosec
                )

        perturbed_words.append(perturbed)

    return " ".join(perturbed_words)


# Load your balanced sample
df = pd.read_json("balanced_test_50.json")

# Add perturbed column
df["perturbed_prompt"] = df["prompt"].apply(perturb_prompt)

# Save with both original + perturbed
df[["prompt", "perturbed_prompt", "type", "classification"]].to_json(
    "balanced_test_50_perturbed.json", orient="records", indent=2
)

print("Perturbation examples:")
for _, row in df.head(3).iterrows():
    print(f"Original:  {row['prompt']}")
    print(f"Perturbed: {row['perturbed_prompt']}")
    print()
