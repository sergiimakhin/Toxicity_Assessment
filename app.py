import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AlbertForSequenceClassification
import os

# --- Config ---
N_FOLDS = 3
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABEL_DISPLAY_NAMES = {
    'toxic': 'Toxic',
    'severe_toxic': 'Severe Toxic',
    'obscene': 'Obscene',
    'threat': 'Threat',
    'insult': 'Insult',
    'identity_hate': 'Identity Hate'
}
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "outputs/tokenizer"
MODEL_BASE_PATH = "outputs/model_fold"

# --- Load Tokenizer ---
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)

tokenizer = load_tokenizer()

# --- Load Models ---
@st.cache_resource
def load_models():
    models = []
    for fold in range(N_FOLDS):
        model_path = f"{MODEL_BASE_PATH}{fold}.pt"
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=len(LABELS))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    return models

models = load_models()

# --- Prediction Function ---
def predict(text):
    encoded = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    fold_probs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for model in models:
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            fold_probs.append(probs)

    avg_probs = np.mean(fold_probs, axis=0).flatten()
    return {label: float(prob) for label, prob in zip(LABELS, avg_probs)}

# --- UI ---
st.title("🧠 Toxic Comment Classifier (ALBERT + Folds)")
st.write("Enter a comment to classify its toxicity using an ensemble of ALBERT models.")

user_input = st.text_area("📝 Comment Text", height=150)

if st.button("🔍 Evaluate toxicity level") and user_input.strip():
    with st.spinner("Predicting..."):
        predictions = predict(user_input)

    # --- Plot: Horizontal Bar Chart ---
    st.subheader("📊 Toxicity level (probabilities in %)")

    percentages = {label: prob * 100 for label, prob in predictions.items()}
    labels = [LABEL_DISPLAY_NAMES[label] for label in percentages.keys()]
    values = list(percentages.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.get_cmap('tab10').colors
    ax.barh(labels, values, color=colors[:len(labels)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Toxicity Prediction per Label")
    st.pyplot(fig)

    # --- Binary Classification ---
    st.subheader("📌 Binary Classification (threshold ≥ 0.5)")
    for label, prob in predictions.items():
        is_toxic = prob >= 0.5
        emoji = "⚠️" if is_toxic else "✅"
        st.write(f"{emoji} **{LABEL_DISPLAY_NAMES[label]}**: {prob:.3f} ({'Toxic' if is_toxic else 'Clean'})")

    st.success("✅ Done!")
