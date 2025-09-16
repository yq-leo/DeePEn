import json
import os


def re_parse_prediction_to_jsonl(path):
    new_lines = []

    # Read and modify lines
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            if "all" in obj:
                obj["prediction"] = obj["all"].split("\nAnswer:")[-1].strip()

            new_lines.append(obj)

    # Overwrite file
    with open(path, "w", encoding="utf-8") as f:
        for obj in new_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    filepath = "res/MMLU/test/InternLM7b+qwen4b/tas3+mas2/ensemble_lr0.15_learning_epochs_nums5.jsonl"
    re_parse_prediction_to_jsonl(filepath)
