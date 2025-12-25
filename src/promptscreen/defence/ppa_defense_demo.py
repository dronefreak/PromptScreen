import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from metrics import AttackEvaluator, AttackResult, MetricsCalculator
from ppa_defence import PolymorphicPromptAssembler
from query_agent import QueryAgent

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    _ = load_dotenv()
    huggingface_token: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

    query_agent: QueryAgent = QueryAgent("tinyllama")

    ppa = PolymorphicPromptAssembler(
        system_prompt="You are a safety guard.", task_topic="safety"
    )
    evaluator: AttackEvaluator = AttackEvaluator("tinyllama", 0.1)
    metrics_calc: MetricsCalculator = MetricsCalculator()

    json_file_path = r"prompt-injection.json"

    with open(json_file_path, encoding="utf-8") as fh:
        data = json.load(fh)
        prompts = [
            entry["prompt"]
            for entry in data
            if entry["classification"] == "prompt-injection"
        ]

        for cnt, prompt in enumerate(prompts):
            secure_prompt, canary = ppa.single_prompt_assemble(prompt)

            start_time = time.time()
            output = query_agent.query(secure_prompt)
            response_time = time.time() - start_time

            attack_result: AttackResult = evaluator.evaluate(
                response_time, output, secure_prompt
            )

            if ppa.leak_detect(output, canary):
                print(f"Leak detected on prompt {cnt}")

            metrics_calc.add_result(attack_result)
            print(f"Prompt {cnt} assessed")

    metrics_calc.evaluate()
