import json
import requests
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
import os

# Configuration Imports 

from config import (
    INPUT_FILE,
    OUTPUT_FILE,
    EMBED_MODEL,
    GEMINI_MODEL,
    GEMINI_API_KEY,
)


#Tool Initialization
try:
   
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f" Error initializing Gemini client: {e}. Check API key.")
    client = None

# Initialize Sentence Transformer Embedder
embedder = SentenceTransformer(EMBED_MODEL)

# Helper Functions

def embedding_similarity(text1, text2):
    """Calculates cosine similarity between two text strings' embeddings."""
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0])

def score_with_gemini(question, gt_answer, rag_answer, rag_retrieved_chunks):

    if client is None:
        return {"correctness": 0, "faithfulness": 0, "hallucination": 0, "explanation": "Gemini Client Not Initialized"}
        
    # Format the RAG's actual retrieved chunks into a single context string
    rag_context = "\n\n".join([c["preview"] for c in rag_retrieved_chunks])
    
    prompt = f"""
    You are evaluating a DrugBank RAG system. Your task is to score the RAG Answer based on three criteria.


    Score using 0–1 scale (e.g., 0.0, 0.5, 1.0).

    1. CORRECTNESS:
        Does the RAG answer match the ground truth answer?
    
    2. FAITHFULNESS:
        Is the RAG answer fully supported by the **RETRIEVED CONTEXT** provided below?
        (Measures if the LLM followed the specific chunks it was given.)

    3. HALLUCINATION:
        Does the RAG answer contain any invented, unsupported, or non-contextual facts **NOT** present in the **RETRIEVED CONTEXT**?
        (1 = no hallucination, 0 = hallucination)

    Return only this JSON object:
    {{
        "correctness": <float>,
        "faithfulness": <float>,
        "hallucination": <float>,
        "explanation": "<short sentence describing the score reasons>"
    }}

    IMPORTANT: Output ONLY the raw JSON object. Do not include markdown, commentary, or extra text. 
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt + f"""

Question: {question}
Ground Truth Answer: {gt_answer}
RAG Answer: {rag_answer}
RETRIEVED CONTEXT: {rag_context}
            """,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # Set a high temperature (low randomness) for consistent judging
                temperature=0.0
            )
        )
        # Parse the JSON response text
        return json.loads(response.text)

    except Exception as e:
        #Catch JSONDecodeError, API errors, etc., and return 0.0 placeholders
        return {
            "correctness": 0.0,
            "faithfulness": 0.0,
            "hallucination": 0.0,
            "explanation": f"Gemini scoring failed: {type(e).__name__} - {str(e)}"
        }

# Main Evaluation Loop
def evaluate():
    results = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scoring RAG Outputs"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping malformed line: {line.strip()}")
            continue

        question = item["question"]
        gt = item["ground_truth_answer"]
        rag = item["rag_answer"]
        

        rag_chunks = item.get("rag_full_output", {}).get("chunks_used", []) 

        # Calculate objective semantic similarity
        sim = embedding_similarity(gt, rag)

        # Score using Gemini (passing the RAG's chunks)
        g = score_with_gemini(question, gt, rag, rag_chunks) 

        # Build the final result object
        out = {
            "drug_name": item.get("drug_name", "N/A"),
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

    # Write all results to the output file
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nEvaluation complete. Saved results for {len(results)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()
