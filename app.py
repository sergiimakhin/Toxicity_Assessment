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
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            continue
        try:
            model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=len(LABELS))
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            models.append(model)
            st.success(f"Loaded model for fold {fold}")
        except Exception as e:
            st.error(f"Error loading model {model_path}: {str(e)}")
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
    with torch.no_grad():
        for i, model in enumerate(models):
            try:
                outputs = model(input_ids, attention_mask=attention_mask)
                if not hasattr(outputs, "logits") or outputs.logits is None:
                    st.error(f"Model {i} did not return logits. Skipping.")
                    continue
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                fold_probs.append(probs)
            except Exception as e:
                st.error(f"Error during inference with model {i}: {str(e)}")

    if not fold_probs:
        st.error("No valid predictions could be made. Check models.")
        return {}

    avg_probs = np.mean(fold_probs, axis=0).flatten()
    return {label: float(prob) for label, prob in zip(LABELS, avg_probs)}

# --- UI ---
st.title("🧠 Toxic Comment Classifier (ALBERT + Folds)")
st.write("Enter a comment to classify its toxicity using an ensemble of ALBERT models.")

user_input = st.text_area("📝 Comment Text", height=150)

if st.button("🔍 Evaluate toxicity level") and user_input.strip():
    with st.spinner("Predicting..."):
        predictions = predict(user_input)

    if predictions:
        # --- Plot: Horizontal Bar Chart ---
        st.subheader("📊 Toxicity level (probabilities in %)")
        percentages = {label: prob * 100 for label, prob in predictions.items()}
        labels = [LABEL_DISPLAY_NAMES[label] for label in percentages.keys()]
        values = list(percentages.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, values, color='skyblue')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Toxicity Prediction per Label")
        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.1f}%", va='center')
        st.pyplot(fig)

        # --- Binary Classification ---
        st.subheader("📌 Binary Classification (threshold ≥ 0.5)")
        for label, prob in predictions.items():
            is_toxic = prob >= 0.5
            emoji = "⚠️" if is_toxic else "✅"
            st.write(f"{emoji} **{LABEL_DISPLAY_NAMES[label]}**: {prob:.3f} ({'Toxic' if is_toxic else 'Clean'})")

        st.success("✅ Done!")
