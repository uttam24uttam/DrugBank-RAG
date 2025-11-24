# import os
# import time
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# # ==========================================
# # CONFIGURATION
# # ==========================================
# DB_FOLDER = "chroma_db_data"
# # We use Qwen 2.5 (1.5B) because it is significantly smarter than the old 1.8B
# MODEL_NAME = "qwen2.5:1.5b"  

# # 1. CHECK DATABASE
# if not os.path.exists(DB_FOLDER):
#     print(f"ERROR: Database folder '{DB_FOLDER}' not found.")
#     print("Please run 'create_vector_db.py' first!")
#     exit()

# # 2. LOAD RESOURCES
# print("Loading Brain (Embeddings)...")
# # Must match the embedding model used during ingestion!
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# print("Connecting to Vector Database...")
# db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)

# print(f"Connecting to Local Model ({MODEL_NAME})...")
# # This connects to the 'ollama serve' process running in your background
# llm = ChatOllama(model=MODEL_NAME, temperature=0)

# # 3. RETRIEVER SETUP
# # k=3 means "Find the top 3 most relevant chunks"
# retriever = db.as_retriever(search_kwargs={"k": 3})

# # 4. RAG PROMPT
# # This instructs Qwen to ONLY use the context provided
# template = """
# You are a precise medical assistant. 
# Answer the user's question using ONLY the context provided below. 
# If the answer is not in the context, simply say "I don't know based on the available data."

# CONTEXT:
# {context}

# QUESTION: 
# {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# # 5. BUILD PIPELINE
# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# # ==========================================
# # CHAT LOOP
# # ==========================================

# print(f"  DRUG RAG SYSTEM ({MODEL_NAME})")

# print("Type 'q' to exit.\n")

# while True:
#     query = input("You: ")
#     if query.lower() in ['q', 'quit', 'exit']:
#         break
    
#     print("Thinking...", end="", flush=True)
#     start_time = time.time()
    
#     try:
#         # This single line runs the whole Search -> Augment -> Generate flow
#         response = rag_chain.invoke(query)
        
#         print(f"\rAI: {response.content}")
#         print(f"\n(Response generated in {time.time() - start_time:.2f}s)")
        
#     except Exception as e:
#         print(f"\nError: {e}")
#         print("Check if Ollama is running via 'ollama serve'!")




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

# ==========================================
# STEP A: ENTITY EXTRACTOR (Solves Problem 1)
# ==========================================
# We ask the LLM to extract the EXACT drug name first so we can filter.
extractor_template = """
Extract the drug name from the user's question. 
Output ONLY the drug name. Nothing else. No intro. No punctuation.
If no drug is mentioned, output "NONE".

User Question: {question}
"""
extractor_chain = ChatPromptTemplate.from_template(extractor_template) | llm | StrOutputParser()

# ==========================================
# STEP B: SYNTHESIS PROMPT (Solves Problem 2 & 3)
# ==========================================
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

# ==========================================
# CHAT LOOP
# ==========================================

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
    
    # --- 2. RETRIEVE WITH FILTER ---
    # This is the MAGIC LINE. We force Chroma to only look at this drug.
    # Note: This requires the drug name in metadata to match exactly. 
    # If the extraction is slightly off ("Metformin" vs "Metformin"), search might fail.
    # For a robust system, we would use fuzzy matching here, but this is a good baseline.
    
    search_kwargs = {"k": 3}
    if drug_name != "NONE":
        # We use the 'contains' logic or exact match depending on your data quality
        # Here we assume the extracted name is close enough to use as a filter
        # If your metadata 'drug_name' is "Metformin HCl", searching "Metformin" might fail with strict $eq
        # So we try standard search first, but if you want strictness:
        # search_kwargs["filter"] = {"drug_name": drug_name} 
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