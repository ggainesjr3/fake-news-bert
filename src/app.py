import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Page Config
st.set_page_config(page_title="Trust Guard AI", page_icon="🛡️", layout="centered")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Update this path to your highest checkpoint number
    path = "./models/results/checkpoint-2000"
    model = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()

st.title("🛡️ Trust Guard AI")
st.subheader("High-Performance Misinformation Audit Terminal")
st.info("Analyze headlines using a fine-tuned BERT Transformer model.")

text_input = st.text_area("Enter Headline or Article Snippet:", placeholder="Paste news text here...", height=150)

if st.button("Analyze Content"):
    if text_input:
        with st.spinner("Analyzing linguistic patterns..."):
            # Tokenize & Predict
            inputs = tokenizer(text_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Convert 'logits' to probabilities (0.0 to 1.0)
                probs = F.softmax(outputs.logits, dim=1)
                confidence, prediction = torch.max(probs, dim=1)
                
            conf_percent = confidence.item() * 100
            is_fake = prediction.item() == 1

            # Professional Display
            st.divider()
            if is_fake:
                st.markdown(f'<div class="result-box" style="background-color: #ffebee; color: #c62828; border: 2px solid #ef5350;">🚨 LIKELY DISINFORMATION</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box" style="background-color: #e8f5e9; color: #2e7d32; border: 2px solid #66bb6a;">✅ VERIFIED NEWS PATTERN</div>', unsafe_allow_html=True)
            
            st.write(f"**Model Confidence:** `{conf_percent:.2f}%`")
            st.progress(conf_percent / 100)
            
            # Explainability logic for the portfolio
            with st.expander("Technical Audit Details"):
                st.write(f"**Model Architecture:** BERT-base-uncased")
                st.write(f"**Classification Logic:** Softmax Probability Distribution")
                st.write("Note: This model analyzes linguistic structures (word choice, tone, and syntax) commonly found in the ISOT dataset.")
    else:
        st.warning("Please enter some text first!")
