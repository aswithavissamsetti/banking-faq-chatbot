import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("https://huggingface.co/spaces/aswitha20/banking-faq-chatbot/raw/main/bank_questions_answers.csv")



# Vectorize questions
tfidf_vectorizer = TfidfVectorizer()
question_vectors = tfidf_vectorizer.fit_transform(df["Question"])

def get_answer(user_query):
    """Find the best matching answer for a user's question."""
    user_vector = tfidf_vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx]["Answer"]




# Gradio Interface
def chatbot_interface(user_query):
    return get_answer(user_query)



iface = gr.Interface(fn=chatbot_interface, inputs="text", outputs="text", title="Banking FAQ Chatbot",
                     description="Ask any banking-related question and get instant answers!")

iface.launch(share=True)
