import argparse
import json
from pathlib import Path
from datasets import load_dataset

def get_nested_text(value):
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    return str(value or "").strip()


def get_nested_score(value, key="toxicity"):
    if not isinstance(value, dict):
        return None
    score = value.get(key)
    if score is None:
        return None
    return float(score)


def preprocess_example(example):
    prompt = example.get("prompt")
    continuation = example.get("continuation")

    return {
        "filename": example.get("filename"),
        "begin": example.get("begin"),
        "end": example.get("end"),
        "challenging": bool(example.get("challenging")),
        "prompt": get_nested_text(prompt),
        "continuation": get_nested_text(continuation),
        "prompt_toxicity": get_nested_score(prompt),
        "continuation_toxicity": get_nested_score(continuation),
    }


def is_valid_example(example):
    return bool(str(example.get("prompt") or "").strip())


def save_jsonl(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(cache_dir, output_dir, max_samples):
    
    raw = load_dataset(
        "allenai/real-toxicity-prompts",
        split="train",
        cache_dir=str(cache_dir),
    )

    processed = raw.map(
        preprocess_example,
        remove_columns=raw.column_names,
        desc="Extracting prompt text and toxicity metadata",
    )
    processed = processed.filter(is_valid_example, desc="Dropping empty prompts")

    if max_samples is not None:
        processed = processed.select(range(min(max_samples, len(processed))))

    challenging = processed.filter(
        lambda example: example["challenging"] is True,
        desc="Selecting challenging prompts",
    )

    save_jsonl(processed, output_dir / "realtoxicity_all.jsonl")
    save_jsonl(challenging, output_dir / "realtoxicity_challenging.jsonl")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evaluation/realtoxicity"),
        help="Directory for processed RealToxicityPrompts JSONL files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/huggingface"),
        help="Directory for Hugging Face downloaded dataset cache.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional small-sample cap for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
