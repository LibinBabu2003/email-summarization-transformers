
# STABILITY FIXES 
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"


# IMPORTS
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# NLTK SAFE SETUP (NO CRASHES)
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab")

setup_nltk()

from nltk.tokenize import sent_tokenize

# LOAD MODEL & TOKENIZER (CPU SAFE)
MODEL_NAME = "facebook/bart-large-cnn"

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map=None,           # disable meta device
        torch_dtype=torch.float32  # force real tensors
    )

    model.to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# STREAMLIT UI
st.set_page_config(
    page_title="Email Summarization System",
    layout="wide"
)

st.title("📧 Transformer-Based Email Summarization System")

st.write(
    """
This application demonstrates **abstractive email summarization** using a pretrained
Transformer model, along with **AI/ML analysis** such as extractive summarization and
sentence-importance visualization.
"""
)

# INPUT
email_text = st.text_area(
    "Paste a long email here:",
    height=300
)

if st.button("Generate Summary") and email_text.strip():

    # TOKENIZATION
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    token_count = inputs["input_ids"].shape[1]

    # ABSTRACTIVE SUMMARY (TRANSFORMER)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=60,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary_text = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    # EXTRACTIVE SUMMARY (TF-IDF)
    sentences = sent_tokenize(email_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    top_indices = np.argsort(sentence_scores)[-3:]
    top_indices = sorted(top_indices)

    extractive_summary = " ".join([sentences[i] for i in top_indices])

    # ATTENTION APPROXIMATION (SIMILARITY BASED)
    all_text = sentences + [summary_text]
    tfidf_vectors = vectorizer.fit_transform(all_text)

    sentence_vectors = tfidf_vectors[:-1]
    summary_vector = tfidf_vectors[-1]

    similarity_scores = cosine_similarity(
        sentence_vectors,
        summary_vector
    ).flatten()

    important_indices = np.argsort(similarity_scores)[-5:]

    # OUTPUT
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Abstractive Summary (Transformer)")
        st.write(summary_text)
        st.markdown(f"**Token count:** {token_count}")

    with col2:
        st.subheader("🔹 Extractive Summary (TF-IDF)")
        st.write(extractive_summary)

    st.subheader("⭐ Important Sentences (Model Focus)")
    for idx in sorted(important_indices):
        st.markdown(f"- {sentences[idx]}")

else:
    st.info("Paste an email and click **Generate Summary**.")
