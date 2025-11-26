import json
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# CONFIGURATION
INPUT_FILE = "chunks/drug_chunks_final.json" 
DB_FOLDER = "chroma_db_data"


if os.path.exists(DB_FOLDER):
    print(f"Deleting old database folder: {DB_FOLDER}")
    shutil.rmtree(DB_FOLDER)

# Loading Chunks
print(f"Loading chunks from {INPUT_FILE}...")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
except FileNotFoundError:
    print(f"ERROR: File not found at {INPUT_FILE}")
    print("Check your folder name or file name.")
    exit()

# Creating Documents
print("Converting chunks to LangChain Documents...")
documents = []

for chunk in raw_chunks:
    doc = Document(
        page_content=chunk["chunk_text"],
        metadata={
            "drug_name": chunk.get("drug_name", chunk.get("name", "Unknown")), 
            "type": chunk.get("type", "general"),
            "section": chunk.get("section", "general")
        }
    )
    documents.append(doc)

print(f"Ready to embed {len(documents)} chunks.")

# Building Database 
print("Loading Model (all-MiniLM-L6-v2)...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Vectorizing and Saving... (This will take a minute)")
db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=DB_FOLDER
)

print("-" * 30)
print(f"SUCCESS! Database created at './{DB_FOLDER}'")

print("-" * 30)
