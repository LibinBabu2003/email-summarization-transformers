---
title: Email Summarization using Transformers
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
---

# 📧 Email Summarization using Transformers

This project demonstrates an **AI/ML-based email summarization system** built using
**Transformer-based pretrained models** from the Hugging Face ecosystem.

## 🚀 Features
- Abstractive email summarization using a Transformer model
- Extractive summarization using TF-IDF (classical NLP baseline)
- Attention-inspired sentence importance visualization
- Token length awareness and compression insight
- Interactive UI built with Streamlit

## 🧠 Model
- Pretrained Transformer model for summarization  
- Fine-tuned and analyzed conceptually for email-style text

## 🖥 Deployment
This application is deployed using **Hugging Face Spaces (Streamlit SDK)** and runs on
free CPU hardware for demonstration and academic purposes.

## 📌 Usage
1. Paste a long email into the text box
2. Click **Generate Summary**
3. View:
   - Abstractive summary
   - Extractive summary
   - Important sentences influencing the summary
