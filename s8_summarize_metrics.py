import json
import pandas as pd

INPUT_FILE = "data/rag_scored.jsonl"
OUTPUT_FILE = "data/rag_summary.csv"

EXPECTED_COLUMNS = [
    "correctness",
    "faithfulness",
    "hallucination",
    "semantic_similarity",
]

def summarize():
    print(f"\nLoading scored results from: {INPUT_FILE}")

    rows = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples.")

    summary = {}
    missing_cols = []

    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            summary[f"mean_{col}"] = df[col].mean()
        else:
            summary[f"mean_{col}"] = None
            missing_cols.append(col)

    summary["num_samples"] = len(df)

    pd.DataFrame([summary]).to_csv(OUTPUT_FILE, index=False)

    print("\n===== Evaluation Summary =====")
    for k, v in summary.items():
        if v is None:
            print(f"{k}: MISSING")
        elif isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("==============================")

    if missing_cols:
        print("\n⚠ WARNING: Some expected metrics were missing from your scored file:")
        for c in missing_cols:
            print(f" - {c}")
        print("Check your scoring script to ensure these metrics are being generated.\n")

if __name__ == "__main__":
    summarize()
