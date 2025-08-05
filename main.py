import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# Load the pre-trained sentiment-analysis model with caching
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# App title and description
st.title("💬 AI-Powered Sentiment Analyzer")
st.markdown("""
This app uses a transformer-based NLP model from HuggingFace 🤗 to detect sentiment in your text.
Enter a sentence below to see if it's **positive**, **negative**, or **neutral**.
""")

# Input field
user_input = st.text_area("✏️ Enter your sentence:", height=150, placeholder="Type something like 'I love this product!'")

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                result = sentiment_pipeline(user_input)[0]
                label = result['label']
                score = result['score'] * 100

                # Output
                if label.upper() == 'POSITIVE':
                    st.success(f"🟢 **Positive** — {score:.2f}% confidence")
                elif label.upper() == 'NEGATIVE':
                    st.error(f"🔴 **Negative** — {score:.2f}% confidence")
                else:
                    st.info(f"⚪ **{label.title()}** — {score:.2f}% confidence")
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter some text before clicking Analyze.")

# Footer
st.markdown("""
---
Made with ❤️ by Jakaria Akash Juwel  
Model: `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace  
""")
