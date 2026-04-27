import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/sentiment-model"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

texts = [
    "I really like this air-gapped AI project.",
    "This project is excellent.",
    "This is terrible and disappointing."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(probs).item()

    print("Input:", text)
    print("Logits:", outputs.logits.tolist())
    print("Probabilities:", probs.tolist())
    print("Prediction ID:", prediction)
    print("Prediction Label:", model.config.id2label[prediction])
    print("---")
