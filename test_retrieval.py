import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# CONFIGURATION
DB_FOLDER = "chroma_db_data"


if not os.path.exists(DB_FOLDER):
    print(f"ERROR: Database folder '{DB_FOLDER}' not found.")
    print("Run 'create_vector_db.py' first!")
    exit()

# Embedding Model
print("Loading Model...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the Database
print("Loading Database...")
db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_function)


# THE SEARCH LOOP
print("\n" + "="*40)
print("   DRUG KNOWLEDGE BASE (SEARCH MODE)")
print("="*40)

while True:
    query = input("\nAsk a question (or 'q' to quit): ")
    if query.lower() == 'q':
        break
    
    #RETRIEVAL STRATEGY 
    # k=3 means " Top 3 best matches"
    results = db.similarity_search(query, k=3)
    
    print(f"\n--- Found {len(results)} Relevant Chunks ---")
    
    for i, doc in enumerate(results):
        drug_name = doc.metadata.get('drug_name', 'Unknown')
        type_ = doc.metadata.get('type', 'General')
        
        print(f"\n[Result {i+1}] Drug: {drug_name} ({type_})")
        print("-" * 30)
    
        print(doc.page_content[:300] + "...") 
        print("-" * 30)