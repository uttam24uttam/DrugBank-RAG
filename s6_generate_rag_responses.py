import json
import requests
from tqdm import tqdm
import pandas as pd
import time
import os


RAG_ENDPOINT = "https://suggestibly-unstrewn-lashaun.ngrok-free.dev/chat"

INPUT_FILE = "data/evaluation_dataset.jsonl"
OUTPUT_FILE = "data/rag_vs_groundtruth.jsonl"

MAX_RETRIES = 4
REQUEST_TIMEOUT = 90          
DELAY_BETWEEN_REQUESTS = 1  
RETRY_BACKOFF = 2             


def query_rag(question: str):
    """Send question to RAG system with retry logic."""
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                RAG_ENDPOINT,
                json={"question": question},
                timeout=REQUEST_TIMEOUT
            )

            if resp.status_code == 200:
                resp_json = resp.json()
                rag_answer = resp_json.get("answer", "")
                return rag_answer, resp_json

            else:
                err = f"Bad status: {resp.status_code}"
                print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {err}")

        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} error: {e}")
            err = str(e)

        sleep_time = RETRY_BACKOFF ** attempt
        print(f"Retrying in {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    return "", {"error": f"FAILED AFTER {MAX_RETRIES} ATTEMPTS: {err}"}


def evaluate():

    print("\nLoading dataset...")
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    print(f"Total samples: {len(lines)}")

    processed = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"\nResuming from existing output file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    processed.add(row["question"])
                except:
                    continue
        print(f"Already processed: {len(processed)} entries\n")
    else:
        print("\nNo previous output found — starting fresh.\n")

    with open(OUTPUT_FILE, "a") as out:

        for line in tqdm(lines):
            entry = json.loads(line)

            question = entry["question"]
            gt_answer = entry["answer"]
            gt_context = entry["ground_truth_context"]
            drug_name = entry["drug_name"]

            if question in processed:
                continue

            rag_answer, rag_raw = query_rag(question)

            result = {
                "drug_name": drug_name,
                "question": question,
                "ground_truth_answer": gt_answer,
                "ground_truth_context": gt_context,
                "rag_answer": rag_answer,
                "rag_full_output": rag_raw
            }

            out.write(json.dumps(result) + "\n")

            time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\nEvaluation Complete!")
    print("Output saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate()
