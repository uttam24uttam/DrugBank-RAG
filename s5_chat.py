import os
from difflib import get_close_matches
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from s4_retrieval import retrieve_candidates, rerank_with_cross_encoder
from BERT_Inference import load_classifier_model, classify_question_type

from config import (
    RETRIEVER_POOL_SIZE,
    TOP_K,
    LLM_MODEL_NAME,
    MASTER_PROMPT_PATH,
    EXTRACTOR_MODEL_NAME,
    EXTRACTOR_USE_FINE_TUNED,
    QUESTION_CLASSIFIER_PATH, 
    LABEL_MAPPING,
)

# LOAD MASTER PROMPT
if not os.path.exists(MASTER_PROMPT_PATH):
    raise FileNotFoundError(f"Master prompt missing: {MASTER_PROMPT_PATH}")

with open(MASTER_PROMPT_PATH, "r") as f:
    MASTER_PROMPT = f.read().strip()


# MODEL LOADER 
def load_chat_model(model_name: str, quantize_4bit=False):
    print(f"[HF Loader] Loading ChatML model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype="auto",
        load_in_4bit=quantize_4bit
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    return model, tokenizer


# LOAD MODELS
print(f"\n[MAIN MODEL] {LLM_MODEL_NAME}")
#For using baseline model
main_model, main_tokenizer = load_chat_model(EXTRACTOR_MODEL_NAME)

extractor_name = EXTRACTOR_MODEL_NAME if EXTRACTOR_USE_FINE_TUNED else LLM_MODEL_NAME
print(f"[EXTRACTOR MODEL] {extractor_name}")
extractor_model, extractor_tokenizer = load_chat_model(extractor_name)

load_classifier_model(QUESTION_CLASSIFIER_PATH, LABEL_MAPPING)


# HELPER
def chatml_generate(model, tokenizer, messages, max_tokens=128):
    """
    messages = [{"role": "system", "content": "..."},
                {"role": "user", "content": "..."}]
    """

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full


# PARSE ASSISTANT RESPONSE
def extract_assistant_text(full_output: str):
    """
    Extract the last <|im_start|>assistant ... <|im_end|> block cleanly.
    """
    if "<|im_start|>assistant" not in full_output:
        return full_output.strip()

    part = full_output.split("<|im_start|>assistant")[-1]
    part = part.split("<|im_end|>")[0]
    return part.strip()


# DRUG NAME EXTRACTOR
def format_extractor_prompt(question: str):
    return [
        {
            "role": "system",
            "content": """You extract ONLY drug names explicitly written in the user's question.

RULES:
- Extract ONLY drug names.
- NEVER answer the user's question.
- NEVER explain.
- Output ONLY the drug name exactly as written.
- If no drug name exists, output exactly: NONE
- Never guess. Never infer.

IMPORTANT:
- Do NOT guess.
- Do NOT infer.
- Do NOT generate a chemical, effect, mechanism, symptom, or anything else.
- Output must be ONLY a drug name from the question or NONE.
- strictly learn from these examples

EXAMPLES:
Q: what are the effects of aspirin?
A: aspirin

Q: what is the chemical formula of aspirin?
A: aspirin

Q: mechanism of acetylsalicylic acid?
A: acetylsalicylic acid

Q: how to deal with constipation?
A: NONE

Q: what should I do when I have fever?
A: NONE

Q: something more about the previous drug?
A: NONE
"""
        },
        {"role": "user", "content": question},
    ]


def extract_drug_name(question: str):
    messages = format_extractor_prompt(question)
    output = chatml_generate(extractor_model, extractor_tokenizer, messages, max_tokens=8)
    extracted = extract_assistant_text(output)

    print("[DEBUG extractor raw]:", repr(extracted))

    clean = extracted.strip().replace('"', "").replace("'", "")
    low = clean.lower()

    if low in ("none", "null", "", "no"):
        return "NONE"

    if len(clean.split()) > 4:
        return "NONE"

    if any(ch in clean for ch in "?:;,"):
        return "NONE"

    if low in {"fever", "pain", "constipation", "headache", "disease"}:
        return "NONE"

    return clean


# BUILD RAG PROMPT FOR MAIN MODEL
def format_rag_prompt(context: str, question: str):
    return [
        {"role": "system", "content": MASTER_PROMPT},
        {
            "role": "assistant",
            "content": f"Here is the available scientific database context:\n\n{context}"
        },
        {"role": "user", "content": question},
    ]


# MAIN RAG PIPELINE
def generate_answer(query: str):
    
    # Calling BERT for question classificatoin
    question_type = classify_question_type(query)

    
    # Returning without hitting RAG if question is not of TypeA
    if question_type == "TYPE_B": 
      
         return {
            "answer": "Sorry, I cannot answer this question. Try asking a different question [BERT] : TYPE_B .",
            "drug_extracted": "N/A - Classification early exit",
            "chunks_used": [],
            "num_chunks": 0
        }

    elif question_type == "TYPE_C": 
      
         return {
            "answer": "Sorry, I cannot answer this question. Try asking a different question [BERT] : TYPE_C .",
            "drug_extracted": "N/A - Classification early exit",
            "chunks_used": [],
            "num_chunks": 0
        }
        
  

    drug = extract_drug_name(query)
    print("[EXTRACTED DRUG] →", drug)

    if drug == "NONE":
        return {
            "answer": "I can only answer drug-related scientific questions.",
            "drug_extracted": "NONE",
            "chunks_used": [],
            "num_chunks": 0
        }

    refined_query = f"{drug}. {query}"

    candidates = retrieve_candidates(refined_query, RETRIEVER_POOL_SIZE)
    if not candidates:
        return {
            "answer": "Data not available.",
            "drug_extracted": drug,
            "chunks_used": [],
            "num_chunks": 0
        }

    reranked = rerank_with_cross_encoder(query, candidates, TOP_K * 2)

    filtered = [d for d in reranked if drug.lower() in d.metadata.get("drug_name", "").lower()]
    final_docs = filtered[:TOP_K] if filtered else reranked[:TOP_K]

    context = "\n\n".join(d.page_content for d in final_docs)

    messages = format_rag_prompt(context, query)

    rendered_prompt = main_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("\n================ FINAL RAG PROMPT SENT TO LLM ================")
    print(rendered_prompt)
    print("===============================================================\n")

    output = chatml_generate(main_model, main_tokenizer, messages, max_tokens=512)


    answer = extract_assistant_text(output)
    print("[DEBUG final answer]:", repr(answer[:200]))

    chunks_used = [
        {
            "drug_name": d.metadata.get("drug_name", "Unknown"),
            "type": d.metadata.get("type", "Unknown"),
            "preview": d.page_content[:200],
        }
        for d in final_docs
    ]

    return {
        "answer": answer,
        "drug_extracted": drug,
        "chunks_used": chunks_used,
        "num_chunks": len(final_docs),
    }


# CLI
if __name__ == "__main__":
    print(f"ChatBaseline Running ({LLM_MODEL_NAME})")
    while True:
        q = input("\nYou: ")
        if q.lower() in {"q", "quit", "exit"}:
            break
        out = generate_answer(q)
        print("AI:", out["answer"])
        print("Drug:", out["drug_extracted"])
