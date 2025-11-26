# Retrieval Settings
RETRIEVER_POOL_SIZE = 20
TOP_K = 5
DB_FOLDER = "chroma_db_data"


# Model Settings
LLM_MODEL_NAME = "utt24/qwen_1.5B_Bayesian"
EXTRACTOR_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
EXTRACTOR_USE_FINE_TUNED = True   #if True → use EXTRACTOR_MODEL_NAME
QUESTION_CLASSIFIER_PATH = "/content/drive/MyDrive/DrugBank-RAG/drug_bank_bert_classifier_final"

# BERT  Lables
LABEL_MAPPING = {
    0: "TYPE_A",
    1: "TYPE_B",
    2: "TYPE_C"
}


# Prompt Settings
MASTER_PROMPT_PATH = "prompts/master_prompt.txt"


# FastAPI Settings 
HOST = "0.0.0.0"
PORT = 8000