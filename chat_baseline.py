# import os
# import time
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from test_retrieval import (
#     retrieve_candidates,
#     rerank_with_cross_encoder
# )

# from config import (
#     RETRIEVER_POOL_SIZE,
#     TOP_K,
#     LLM_MODEL_NAME,
#     MASTER_PROMPT_PATH,
# )


# # Load master prompt
# if not os.path.exists(MASTER_PROMPT_PATH):
#     raise FileNotFoundError(f"Master prompt file missing: {MASTER_PROMPT_PATH}")

# with open(MASTER_PROMPT_PATH, "r") as f:
#     MASTER_PROMPT = f.read()

# # Initialize LLM
# llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

# # Create prompt template
# rag_prompt = ChatPromptTemplate.from_template(MASTER_PROMPT)


# # ======================================================
# # HELPER: Run full RAG pipeline
# # ======================================================
# def generate_answer(query: str):
#     """
#     Full RAG pipeline:
#     1. Retrieve candidate chunks (bi-encoder)
#     2. Rerank with cross-encoder
#     3. Build final prompt
#     4. LLM generates final answer
#     """

#     # ---------- 1. Stage 1 Retrieval ----------
#     candidates = retrieve_candidates(query, RETRIEVER_POOL_SIZE)

#     # ---------- 2. Stage 2 Reranking ----------
#     final_docs = rerank_with_cross_encoder(query, candidates, TOP_K)

#     # Build context
#     context_text = "\n\n".join([d.page_content for d in final_docs])

#     # ---------- 3. Final LLM Stage ----------
#     chain = rag_prompt | llm | StrOutputParser()
#     response = chain.invoke({"context": context_text, "question": query})

#     # Logging metadata
#     used_chunks = [
#         {
#             "drug_name": d.metadata.get("drug_name", "Unknown"),
#             "type": d.metadata.get("type", "Unknown"),
#             "preview": d.page_content[:200],
#         }
#         for d in final_docs
#     ]

#     return {
#         "answer": response,
#         "chunks_used": used_chunks,
#         "num_chunks": len(final_docs),
#     }


# # ======================================================
# # CLI MODE (Optional)
# # ======================================================
# if __name__ == "__main__":
#     print(f" ChatBaseline (Model: {LLM_MODEL_NAME})")

#     while True:
#         q = input("\nYou: ")
#         if q.lower() in ["q", "quit"]:
#             break

#         print("Thinking...")

#         result = generate_answer(q)

#         print("\nAI:", result["answer"])
#         print("\n[Debug Chunks Used]")
#         for c in result["chunks_used"]:
#             print(f"- {c['drug_name']} | {c['type']} | {c['preview']}...")












# import os
# import time
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from test_retrieval import (
#     retrieve_candidates,
#     rerank_with_cross_encoder
# )

# from config import (
#     RETRIEVER_POOL_SIZE,
#     TOP_K,
#     LLM_MODEL_NAME,
#     MASTER_PROMPT_PATH,
# )


# # ==========================================
# # LOAD MASTER PROMPT
# # ==========================================
# if not os.path.exists(MASTER_PROMPT_PATH):
#     raise FileNotFoundError(f"Master prompt file missing: {MASTER_PROMPT_PATH}")

# with open(MASTER_PROMPT_PATH, "r") as f:
#     MASTER_PROMPT = f.read()

# rag_prompt = ChatPromptTemplate.from_template(MASTER_PROMPT)

# # ==========================================
# # LOAD LLM
# # ==========================================
# llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)


# # ==========================================
# # DRUG NAME EXTRACTOR (RESTORED)
# # ==========================================
# extractor_template = """
# Extract the drug name from the user's question. 
# Output ONLY the drug name, without punctuation.
# If no drug is mentioned, output "NONE".

# User Question: {question}
# """
# extractor_chain = (
#     ChatPromptTemplate.from_template(extractor_template)
#     | llm
#     | StrOutputParser()
# )


# # ==========================================
# # FULL RAG PIPELINE (FIXED)
# # ==========================================
# def generate_answer(query: str):

#     # ---------- 1. Extract Drug Name ----------
#     drug_name = extractor_chain.invoke({"question": query}).strip()
#     drug_name = drug_name.replace('"', "").replace("'", "")

#     # ---------- 2. Refine Query ----------
#     refined_query = f"Information about drug {drug_name}. User query: {query}"

#     # ---------- 3. Stage 1 Retrieval (pool-size) ----------
#     candidates = retrieve_candidates(refined_query, RETRIEVER_POOL_SIZE)

#     # ---------- 4. Drug Filtering ----------
#     filtered = []
#     for doc in candidates:
#         doc_name = doc.metadata.get("drug_name", "").lower()
#         if drug_name.lower() in doc_name or drug_name == "NONE":
#             filtered.append(doc)

#     # Fallback: If filtering removed everything, keep original candidates
#     if len(filtered) == 0:
#         filtered = candidates

#     # ---------- 5. Stage 2 Reranking ----------
#     final_docs = rerank_with_cross_encoder(query, filtered, TOP_K)

#     # ---------- 6. Build Context ----------
#     context_text = "\n\n".join([d.page_content for d in final_docs])

#     # ---------- 7. Final LLM Response ----------
#     chain = rag_prompt | llm | StrOutputParser()
#     response = chain.invoke({"context": context_text, "question": query})

#     # ---------- 8. Logging ----------
#     used_chunks = [
#         {
#             "drug_name": d.metadata.get("drug_name", "Unknown"),
#             "type": d.metadata.get("type", "Unknown"),
#             "preview": d.page_content[:200],
#         }
#         for d in final_docs
#     ]

#     return {
#         "answer": response,
#         "drug_extracted": drug_name,
#         "chunks_used": used_chunks,
#         "num_chunks": len(final_docs),
#     }


# # ==========================================
# # CLI MODE
# # ==========================================
# if __name__ == "__main__":
#     print(f" ChatBaseline (Model: {LLM_MODEL_NAME})")

#     while True:
#         q = input("\nYou: ")
#         if q.lower() in ["q", "quit"]:
#             break

#         print("Thinking...")

#         out = generate_answer(q)
#         print("\nAI:", out["answer"])
#         print(f"[Extracted drug: {out['drug_extracted']}]")
#         print("[Chunks Used]")
#         for c in out["chunks_used"]:
#             print(f"- {c['drug_name']} | preview: {c['preview'][:80]}...")












# import os
# import time
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from test_retrieval import (
#     retrieve_candidates,
#     rerank_with_cross_encoder
# )

# from config import (
#     RETRIEVER_POOL_SIZE,
#     TOP_K,
#     LLM_MODEL_NAME,
#     MASTER_PROMPT_PATH,
# )

# # ==========================================
# # LOAD MASTER PROMPT
# # ==========================================
# if not os.path.exists(MASTER_PROMPT_PATH):
#     raise FileNotFoundError(f"Master prompt file missing: {MASTER_PROMPT_PATH}")

# with open(MASTER_PROMPT_PATH, "r") as f:
#     MASTER_PROMPT = f.read()

# rag_prompt = ChatPromptTemplate.from_template(MASTER_PROMPT)

# # ==========================================
# # LOAD LLM
# # ==========================================
# llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

# # ==========================================
# # STRONG DRUG EXTRACTOR (FIXED)
# # ==========================================
# extractor_template = """
# You are a drug-name extractor for a pharmacology RAG system.

# Extract ONLY the drug name mentioned in the user question.

# RULES:
# - Output ONLY the drug name (no extra words).
# - Preserve the full multi-word drug (e.g., "magnesium acetate tetrahydrate").
# - If multiple drugs appear, return the **most specific one**.
# - If no drug appears, return: NONE

# Examples:
# Q: "Tell me about metformin"
# A: metformin

# Q: "Mechanism of magnesium acetate tetrahydrate"
# A: magnesium acetate tetrahydrate

# Q: "Is Tylenol safe?"
# A: tylenol

# Q: "Hi"
# A: NONE

# User Question: {question}
# """

# extractor_chain = (
#     ChatPromptTemplate.from_template(extractor_template)
#     | llm
#     | StrOutputParser()
# )

# # ==========================================
# # FULL RAG PIPELINE (FULLY FIXED)
# # ==========================================
# def generate_answer(query: str):

#     # ---------- 1. Extract Drug Name ----------
#     drug_name = extractor_chain.invoke({"question": query}).strip()
#     drug_name = drug_name.replace('"', "").replace("'", "")

#     # ---------- 2. Refine Query (only if extractor succeeded) ----------
#     if drug_name != "NONE":
#         refined_query = f"{drug_name}. {query}"
#     else:
#         refined_query = query

#     # ---------- 3. Stage 1 Retrieval (large pool) ----------
#     candidates = retrieve_candidates(refined_query, RETRIEVER_POOL_SIZE)

#     if not candidates:
#         return {
#             "answer": "Data not available.",
#             "drug_extracted": drug_name,
#             "chunks_used": [],
#             "num_chunks": 0,
#         }

#     # ---------- 4. Stage 2 Reranking ----------
#     reranked = rerank_with_cross_encoder(query, candidates, TOP_K * 2)

#     # ---------- 5. Drug Filtering AFTER reranking ----------
#     filtered = []
#     for doc in reranked:
#         doc_name = doc.metadata.get("drug_name", "").lower()
#         if drug_name != "NONE" and drug_name.lower() in doc_name:
#             filtered.append(doc)

#     # fallback if filtering removed too much
#     if len(filtered) == 0:
#         final_docs = reranked[:TOP_K]
#     else:
#         final_docs = filtered[:TOP_K]

#     # ---------- 6. Build Context ----------
#     context_text = "\n\n".join([d.page_content for d in final_docs])

#     # ---------- DEBUG: print final prompt given to LLM ----------
#     final_prompt = rag_prompt.format(context=context_text, question=query)
#     print("\n==================== FINAL PROMPT SENT TO LLM ====================")
#     print(final_prompt)
#     print("=================================================================\n")

#     # ---------- 7. Final LLM Response ----------
#     chain = rag_prompt | llm | StrOutputParser()
#     response = chain.invoke({"context": context_text, "question": query})

#     # ---------- 8. Logging ----------
#     used_chunks = [
#         {
#             "drug_name": d.metadata.get("drug_name", "Unknown"),
#             "type": d.metadata.get("type", "Unknown"),
#             "preview": d.page_content[:200],
#         }
#         for d in final_docs
#     ]

#     return {
#         "answer": response,
#         "drug_extracted": drug_name,
#         "chunks_used": used_chunks,
#         "num_chunks": len(final_docs),
#     }

# # ==========================================
# # CLI MODE
# # ==========================================
# if __name__ == "__main__":
#     print(f" ChatBaseline (Model: {LLM_MODEL_NAME})")

#     while True:
#         q = input("\nYou: ")
#         if q.lower() in ["q", "quit"]:
#             break

#         print("Thinking...")

#         out = generate_answer(q)
#         print("\nAI:", out["answer"])
#         print(f"[Extracted drug: {out['drug_extracted']}]")
#         print("[Chunks Used]")
#         for c in out["chunks_used"]:
#             print(f"- {c['drug_name']} | preview: {c['preview'][:80]}...")





# import os
# import time
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from test_retrieval import (
#     retrieve_candidates,
#     rerank_with_cross_encoder
# )

# from config import (
#     RETRIEVER_POOL_SIZE,
#     TOP_K,
#     LLM_MODEL_NAME,
#     MASTER_PROMPT_PATH,
# )

# # ==========================================
# # LOAD MASTER PROMPT
# # ==========================================
# if not os.path.exists(MASTER_PROMPT_PATH):
#     raise FileNotFoundError(f"Master prompt file missing: {MASTER_PROMPT_PATH}")

# with open(MASTER_PROMPT_PATH, "r") as f:
#     MASTER_PROMPT = f.read()

# rag_prompt = ChatPromptTemplate.from_template(MASTER_PROMPT)

# # ==========================================
# # LOAD LLM
# # ==========================================
# llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

# # ==========================================
# # DRUG NAME EXTRACTOR
# # ==========================================
# extractor_template = """
# You are a drug-name extractor for a pharmacology RAG system.

# Extract ONLY the drug name mentioned in the user question.
# only extract a drug name if explicitly in the question.

# RULES:
# - Output ONLY the drug name — no explanations, no punctuation.
# - Preserve multi-word drug names (e.g., "magnesium acetate tetrahydrate").
# - If multiple drugs appear, return the most specific.
# - If no drug appears, output: NONE

# User Question: {question}
# """

# extractor_chain = (
#     ChatPromptTemplate.from_template(extractor_template)
#     | llm
#     | StrOutputParser()
# )

# # ==========================================
# # FULL RAG PIPELINE
# # ==========================================
# def generate_answer(query: str):

#     # ---------- 1. Extract Drug Name ----------
#     drug_name = extractor_chain.invoke({"question": query}).strip()
#     drug_name = drug_name.replace('"', "").replace("'", "")

#     # ---------- 1B. If NOT a drug question → skip RAG ----------
#     if drug_name == "NONE":
#         final_prompt_debug = f"""
# ===== FINAL MASTER PROMPT SENT TO LLM =====

# (NONE — skipped RAG)

# User asked: {query}
# Model response: I can only answer drug-related scientific questions.

# ===========================================
# """
#         print(final_prompt_debug)

#         return {
#             "answer": "I can only answer drug-related scientific questions.",
#             "drug_extracted": "NONE",
#             "chunks_used": [],
#             "num_chunks": 0,
#         }

#     # ---------- 2. Build refined retrieval query ----------
#     refined_query = f"{drug_name}. {query}"

#     # ---------- 3. Stage 1 Retrieval ----------
#     candidates = retrieve_candidates(refined_query, RETRIEVER_POOL_SIZE)

#     if not candidates:
#         return {
#             "answer": "Data not available.",
#             "drug_extracted": drug_name,
#             "chunks_used": [],
#             "num_chunks": 0,
#         }

#     # ---------- 4. Stage 2 Reranking ----------
#     reranked = rerank_with_cross_encoder(query, candidates, TOP_K * 2)

#     # ---------- 5. Filter docs for extracted drug ----------
#     filtered = []
#     for doc in reranked:
#         name = doc.metadata.get("drug_name", "").lower()
#         if drug_name.lower() in name:
#             filtered.append(doc)

#     final_docs = filtered[:TOP_K] if filtered else reranked[:TOP_K]

#     # ---------- 6. Build context ----------
#     context_text = "\n\n".join([d.page_content for d in final_docs])

#     # ---------- 7. FINAL PROMPT DEBUGGING ----------
#     final_prompt_text = MASTER_PROMPT.replace("{context}", context_text).replace("{question}", query)

#     print("\n\n===== FINAL MASTER PROMPT SENT TO LLM =====")
#     print(final_prompt_text)
#     print("===========================================\n\n")

#     # ---------- 8. Final LLM call ----------
#     chain = rag_prompt | llm | StrOutputParser()
#     response = chain.invoke({"context": context_text, "question": query})

#     # ---------- 9. Logging ----------
#     used_chunks = [
#         {
#             "drug_name": d.metadata.get("drug_name", "Unknown"),
#             "type": d.metadata.get("type", "Unknown"),
#             "preview": d.page_content[:200],
#         }
#         for d in final_docs
#     ]

#     return {
#         "answer": response,
#         "drug_extracted": drug_name,
#         "chunks_used": used_chunks,
#         "num_chunks": len(final_docs),
#     }

# # ==========================================
# # CLI MODE
# # ==========================================
# if __name__ == "__main__":
#     print(f" ChatBaseline (Model: {LLM_MODEL_NAME})")

#     while True:
#         q = input("\nYou: ")
#         if q.lower() in ["q", "quit"]:
#             break

#         print("Thinking...")
#         out = generate_answer(q)

#         print("\nAI:", out["answer"])
#         print(f"[Extracted drug: {out['drug_extracted']}]")
#         print("[Chunks Used]")
#         for c in out["chunks_used"]:
#             print(f"- {c['drug_name']} | preview: {c['preview'][:80]}...")





import os
from difflib import get_close_matches

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrieval pipeline
from test_retrieval import (
    retrieve_candidates,
    rerank_with_cross_encoder
)

# Config
from config import (
    RETRIEVER_POOL_SIZE,
    TOP_K,
    LLM_MODEL_NAME,
    MASTER_PROMPT_PATH,
    EXTRACTOR_MODEL_NAME,
    EXTRACTOR_USE_FINE_TUNED,
)

# ============================================================
# HF MODEL LOADER
# ============================================================
def load_hf_model(model_name: str, quantize_4bit: bool = False):
    """
    Loads a HuggingFace causal language model via LangChain pipeline.
    quantize_4bit=True -> loads in 4-bit (requires bitsandbytes).
    """
    print(f"\n[HF Loader] Loading model: {model_name}")

    if quantize_4bit:
        print("[HF Loader] Using 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.0001,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=generator)


# ============================================================
# LOAD MASTER RAG PROMPT
# ============================================================
if not os.path.exists(MASTER_PROMPT_PATH):
    raise FileNotFoundError(f"Master prompt missing: {MASTER_PROMPT_PATH}")

with open(MASTER_PROMPT_PATH, "r") as f:
    MASTER_PROMPT = f.read()

rag_prompt = ChatPromptTemplate.from_template(MASTER_PROMPT)


# ============================================================
# LOAD LLMs — MAIN + EXTRACTOR
# ============================================================
print(f"\n[MAIN MODEL] Loading FINAL generator model:")
print(f" LLM_MODEL_NAME = {LLM_MODEL_NAME}")
llm_main = load_hf_model(LLM_MODEL_NAME, quantize_4bit=False)

# Determine extractor model
if EXTRACTOR_USE_FINE_TUNED:
    extractor_model_to_use = EXTRACTOR_MODEL_NAME
    print(f"\n[EXTRACTOR MODE] Using SEPARATE extractor model (flag=True):")
    print(f" EXTRACTOR_MODEL_NAME = {EXTRACTOR_MODEL_NAME}")
else:
    extractor_model_to_use = LLM_MODEL_NAME
    print(f"\n[EXTRACTOR MODE] Using SAME model as generator (flag=False):")
    print(f" extractor_model_to_use = LLM_MODEL_NAME = {LLM_MODEL_NAME}")

print(f"\n[EXTRACTOR MODEL] Loading extractor model:")
print(f" {extractor_model_to_use}")
llm_extractor = load_hf_model(extractor_model_to_use)

print(llm_extractor.pipeline.tokenizer.special_tokens_map)
print(llm_extractor.pipeline.tokenizer.chat_template)



# ============================================================
# LOAD DRUG NAME LIST (OPTIONAL — FUZZY FALLBACK)
# ============================================================
DRUG_NAMES = []
drug_list_path = "data/drug_names.txt"

if os.path.exists(drug_list_path):
    with open(drug_list_path, "r", encoding="utf-8") as fh:
        DRUG_NAMES = [line.strip() for line in fh if line.strip()]
    print(f"[Drug List] Loaded {len(DRUG_NAMES)} drug names for fuzzy matching")
else:
    print("[Drug List] WARNING: drug_names.txt not found — no fuzzy fallback")


# ============================================================
# DRUG EXTRACTION CHAIN
# ============================================================
extractor_template = """
You are a drug-name extractor for a pharmacology RAG system.

Extract ONLY the drug name explicitly mentioned in the user question.

RULES:
- Output ONLY the drug name — no extra text.
- Preserve multi-word drug names.
- If multiple drugs occur, return the most specific.
- If NO drug is present, return: NONE

User Question: {question}
"""

extractor_chain = (
    ChatPromptTemplate.from_template(extractor_template)
    | llm_extractor
    | StrOutputParser()
)


# ============================================================
# FULL RAG PIPELINE
# ============================================================
def generate_answer(query: str):
    # ---- 1. Extract drug name ----
    drug_name = extractor_chain.invoke({"question": query}).strip()
    drug_name = drug_name.replace('"', "").replace("'", "")

    # ---- Fuzzy fallback if NONE ----
    if drug_name.upper() == "NONE":
        if DRUG_NAMES:
            candidates = get_close_matches(query.lower(), DRUG_NAMES, n=3, cutoff=0.75)
            if candidates:
                print(f"[Fuzzy] Matched: {candidates[0]}")
                drug_name = candidates[0]
            else:
                return {
                    "answer": "I can only answer drug-related scientific questions.",
                    "drug_extracted": "NONE",
                    "chunks_used": [],
                    "num_chunks": 0,
                }
        else:
            return {
                "answer": "I can only answer drug-related scientific questions.",
                "drug_extracted": "NONE",
                "chunks_used": [],
                "num_chunks": 0,
            }

    # ---- 2. Refine query ----
    refined_query = f"{drug_name}. {query}"

    # ---- 3. Stage 1 Retrieval ----
    candidates = retrieve_candidates(refined_query, RETRIEVER_POOL_SIZE)

    if not candidates:
        return {
            "answer": "Data not available.",
            "drug_extracted": drug_name,
            "chunks_used": [],
            "num_chunks": 0,
        }

    # ---- 4. Stage 2 Reranking ----
    reranked = rerank_with_cross_encoder(query, candidates, TOP_K * 2)

    # ---- 5. Filter by drug ----
    filtered = [
        d for d in reranked
        if drug_name.lower() in d.metadata.get("drug_name", "").lower()
    ]

    final_docs = filtered[:TOP_K] if filtered else reranked[:TOP_K]

    # ---- 6. Build context ----
    context_text = "\n\n".join([d.page_content for d in final_docs])

    # ---- 7. Debug — print full RAG prompt ----
    final_prompt_for_llm = MASTER_PROMPT.replace("{context}", context_text).replace("{question}", query)
    print("\n===== FINAL PROMPT TO LLM =====")
    print(final_prompt_for_llm)
    print("================================\n")

    # ---- 8. Final Answer from Fine-Tuned Model ----
    chain = rag_prompt | llm_main | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})

    # ---- 9. Logging chunks ----
    used_chunks = [
        {
            "drug_name": d.metadata.get("drug_name", "Unknown"),
            "type": d.metadata.get("type", "Unknown"),
            "preview": d.page_content[:200],
        }
        for d in final_docs
    ]

    return {
        "answer": response,
        "drug_extracted": drug_name,
        "chunks_used": used_chunks,
        "num_chunks": len(final_docs),
    }




# ============================================================
# CLI MODE
# ============================================================
if __name__ == "__main__":
    print(f"\nChatBaseline Active (Main: {LLM_MODEL_NAME} | Extractor: {extractor_model_to_use})")

    while True:
        q = input("\nYou: ")
        if q.lower() in ["q", "quit", "exit"]:
            break

        print("Thinking...")
        out = generate_answer(q)

        print("\nAI:", out["answer"])
        print(f"[Extracted drug: {out['drug_extracted']}]")
        print("[Chunks Used]")
        for c in out["chunks_used"]:
            print(f"- {c['drug_name']} | {c['preview'][:80]}...")
