import json
import requests
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types

RAG_ENDPOINT = "https://suggestibly-unstrewn-lashaun.ngrok-free.dev/chat"
INPUT_FILE = "data/rag_vs_groundtruth.jsonl"
OUTPUT_FILE = "data/rag_scored.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_API_KEY = "AIzaSyBH2nb5HF3U5cdALJvx_xx5OEz0Eu0QWVw"
GEMINI_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

def query_rag(question):
    try:
        resp = requests.post(
            RAG_ENDPOINT,
            json={"question": question},
            timeout=20
        )
        return resp.json()
    except:
        return {"answer": ""}

def embedding_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0])

def score_with_gemini(question, gt_answer, rag_answer, gt_context):
    prompt = f"""
    You are evaluating a DrugBank RAG system.

    Score using 0–1 scale:

    1. CORRECTNESS:
       Does the RAG answer match the ground truth answer?
    
    2. FAITHFULNESS:
       Is the RAG answer fully supported by the ground truth context?

    3. HALLUCINATION:
       Does the RAG answer contain any invented, unsupported, or non-contextual facts?
       (1 = no hallucination, 0 = hallucination)

    Return only this JSON:
    {{
        "correctness": <float>,
        "faithfulness": <float>,
        "hallucination": <float>,
        "explanation": "<short sentence>"
    }}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt + f"""

Question: {question}
Ground Truth Answer: {gt_answer}
RAG Answer: {rag_answer}
Ground Truth Context: {json.dumps(gt_context)}
            """,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)

    except Exception as e:
        return {
            "correctness": 0,
            "faithfulness": 0,
            "hallucination": 0,
            "explanation": str(e)
        }

def evaluate():
    results = []

    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)

        question = item["question"]
        gt = item["ground_truth_answer"]
        rag = item["rag_answer"]
        gt_context = item["ground_truth_context"]

        sim = embedding_similarity(gt, rag)

        g = score_with_gemini(question, gt, rag, gt_context)

        out = {
            "drug_name": item["drug_name"],
            "question": question,
            "ground_truth": gt,
            "rag_answer": rag,

            "semantic_similarity": sim,

            "correctness": g["correctness"],
            "faithfulness": g["faithfulness"],
            "hallucination": g["hallucination"],
            "explanation": g["explanation"]
        }

        results.append(out)

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("Saved evaluation to", OUTPUT_FILE)

if __name__ == "__main__":
    evaluate()
