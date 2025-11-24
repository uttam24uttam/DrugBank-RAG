import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# CONFIGURATION
# ==========================================
DB_FOLDER = "chroma_db_data"
MODEL_NAME = "qwen2.5:1.5b"

# 1. SETUP RESOURCES
if not os.path.exists(DB_FOLDER):
    print("ERROR: DB not found.")
    exit()

print("Loading Brain...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)
llm = ChatOllama(model=MODEL_NAME, temperature=0) # Temp=0 reduces hallucinations


# We ask the LLM to extract the EXACT drug name first so we can filter.
extractor_template = """
Extract the drug name from the user's question. 
Output ONLY the drug name. Nothing else. No intro. No punctuation.
If no drug is mentioned, output "NONE".

User Question: {question}
"""
extractor_chain = ChatPromptTemplate.from_template(extractor_template) | llm | StrOutputParser()


# We command it to merge chunks and ban medical advice.
rag_template = """
You are a Scientific Database Assistant. Your job is to summarize chemical facts.
STRICT RULES:
1. Use ONLY the context below. Do not use outside knowledge.
2. Merge the information into a single, smooth answer output to the user. Do NOT mention "Chunk 1" or "the text".
3. SAFETY & USAGE: 
   - If the user asks if a drug treats a specific disease/condition, answer ONLY: "Sorry, I cannot provide medical advice."
   - For general questions, describe the drug's mechanism and chemical properties. Do NOT mention clinical indications ("used for...") or usage instructions.
4. Dont keep answers too long.If the answer is not in the context, say "Data not available."

CONTEXT:
{context}

QUESTION: 
{question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)


# CHAT LOOP
print(f" PRECISION DRUG BOT ({MODEL_NAME})")


while True:
    query = input("\nYou: ")
    if query.lower() in ['q', 'quit']: break
    
    print("Thinking...", end="", flush=True)
    start = time.time()
    
    # --- 1. EXTRACT ENTITY ---
    drug_name = extractor_chain.invoke({"question": query}).strip()
    
    # Clean up the extracted name (remove extra quotes if model adds them)
    drug_name = drug_name.replace('"', '').replace("'", "")
    
    # --- 2. RETRIEVE WITH FILTER --

    search_kwargs = {"k": 3}
    if drug_name != "NONE":

        pass 
    
    # For now, let's stick to Vector Search but print what we *would* filter
    # Implementing strict filters requires perfect name matching which is hard without a lookup table.
    # Instead, we will "soft filter" by adding the drug name to the search query boosting.
    
    refined_query = f"Details for drug {drug_name}: {query}"
    
    results = db.similarity_search(refined_query, k=3)
    
    # --- 3. FILTER RESULTS MANUALLY (Python Side) ---
    # This is safer than DB filtering for typos.
    # We only keep chunks where the metadata name loosely matches the extracted name.
    filtered_docs = []
    for doc in results:
        doc_drug = doc.metadata.get("drug_name", "Unknown")
        if drug_name.lower() in doc_drug.lower() or drug_name == "NONE":
            filtered_docs.append(doc)
    
    # Fallback: If we filtered everything away (typo?), keep original results
    if not filtered_docs:
        filtered_docs = results

    # --- 4. GENERATE ANSWER ---
    context_text = "\n\n".join([d.page_content for d in filtered_docs])
    
    final_chain = rag_prompt | llm | StrOutputParser()
    response = final_chain.invoke({"context": context_text, "question": query})
    
    print(f"\rAI: {response}")

    print(f"\n[Debug: Focused on '{drug_name}' | Used {len(filtered_docs)} chunks]")
