import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("https://huggingface.co/spaces/aswitha20/banking-faq-chatbot/raw/main/bank_questions_answers.csv")

# Vectorize questions
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    strip_accents='unicode'
)
question_vectors = tfidf_vectorizer.fit_transform(df["Question"])

def get_answer(user_query):
    """Find the best matching answer for a user's question."""
    user_vector = tfidf_vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = np.argmax(similarities)

    # Handle low similarity matches
    if similarities[0][best_match_idx] < 0.2:
        return "Sorry, I couldn't find a relevant answer. Could you rephrase your question?"
    
    return df.iloc[best_match_idx]["Answer"]

# Gradio Interface
iface = gr.Interface(
    fn=get_answer,
    inputs="text",
    outputs="text",
    title="Banking FAQ Chatbot",
    description="Ask any banking-related question and get instant answers!"
)

iface.launch(share=True)
    
