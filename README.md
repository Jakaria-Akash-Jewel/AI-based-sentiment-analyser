💬 AI-Powered Sentiment Analyzer
This project is a simple yet powerful sentiment analysis web app built with Streamlit and a pre-trained transformer model from HuggingFace 🤗.
It allows users to input any sentence and instantly receive a sentiment prediction—Positive, Negative, or Neutral—along with a confidence score.

🚀 Features
🧠 Uses the distilbert-base-uncased-finetuned-sst-2-english model
📊 Provides real-time sentiment analysis with confidence score
💻 Easy-to-use web interface
⚡ Fast performance with caching
❤️ Clean design with user-friendly feedback

🛠️ How It Works
The user types a sentence into the input box.
Upon clicking "Analyze Sentiment", the app uses a HuggingFace pipeline to process the text.
The model returns the sentiment and a confidence score.

The result is shown with clear color-coded feedback:
🟢 Positive, 🔴 Negative, ⚪ Neutral

🔧 Tech Stack
Python 🐍
Streamlit
HuggingFace Transformers
distilBERT NLP model

