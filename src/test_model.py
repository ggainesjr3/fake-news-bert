import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Load the model we just trained
# Note: Check your models/results folder to find the exact "checkpoint" folder name
model_path = './models/results/checkpoint-2000' # Change '2000' to your highest number
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_news(text):
    # Prepare the text for BERT
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Get the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "FAKE" if prediction == 1 else "REAL"

# 3. Test it out!
print("--- BERT Fake News Detector ---")
while True:
    user_input = input("\nPaste a news headline (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    
    result = predict_news(user_input)
    print(f"Prediction: {result}")
