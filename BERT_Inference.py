# BERT_Inference.py

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import os

CLASSIFIER_MODEL = None
CLASSIFIER_TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier_model(model_path: str, label_map: dict):

    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER

    if CLASSIFIER_MODEL is None:
        if not os.path.exists(model_path):
             print(f"[ERROR] Model path not found: {model_path}")
             raise FileNotFoundError(f"BERT model files missing from: {model_path}")
             
        print(f"[BERT CLASSIFIER] Loading model from: {model_path} on device: {DEVICE}")
        
        # Load Tokenizer
        CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        
        # Load Model 
        CLASSIFIER_MODEL = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label_map),
            id2label=label_map, 
        ).to(DEVICE)
        
        CLASSIFIER_MODEL.eval() 
        print("[BERT CLASSIFIER] Model loaded successfully.")

    return CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER


# Inference Function
def classify_question_type(query: str):
    model, tokenizer = CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER
    
    if model is None or tokenizer is None:
        # This occurs if load_classifier_model was not called first.
        print("[ERROR] BERT Classifier not initialized. Cannot classify.")
        return "UNKNOWN_ERROR"


    inputs = tokenizer(
        query, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(DEVICE)

    
    with torch.no_grad():
        outputs = model(**inputs)
        
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    
    predicted_label = model.config.id2label.get(predicted_class_id, "UNKNOWN_LABEL")
    
    return predicted_label