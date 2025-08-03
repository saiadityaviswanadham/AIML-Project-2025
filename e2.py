import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM
)
import streamlit as st
import logging
import time
import psutil
import sqlite3
import os
from datetime import datetime

# ------------ ENVIRONMENT & LOGGING ------------
logging.basicConfig(filename='viswamodel_chatbot.log', level=logging.INFO, format='%(asctime)s %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ DATABASE SETUP -------------------
def init_db():
    conn = sqlite3.connect("viswamodel_users.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        user_id TEXT,
                        timestamp TEXT,
                        user_message TEXT,
                        bot_response TEXT,
                        sentiment TEXT,
                        emotion TEXT,
                        confidence REAL
                    )''')
    conn.commit()
    conn.close()
init_db()

# ------------ LOAD MODELS ----------------------
# Sentiment models
sentiment1_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment1_tokenizer = AutoTokenizer.from_pretrained(sentiment1_name)
sentiment1_model = AutoModelForSequenceClassification.from_pretrained(sentiment1_name).to(DEVICE)
sentiment1_pipe = pipeline("sentiment-analysis", model=sentiment1_model, tokenizer=sentiment1_tokenizer, device=0 if torch.cuda.is_available() else -1)

sentiment2_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment2_tokenizer = AutoTokenizer.from_pretrained(sentiment2_name)
sentiment2_model = AutoModelForSequenceClassification.from_pretrained(sentiment2_name).to(DEVICE)
sentiment2_pipe = pipeline("sentiment-analysis", model=sentiment2_model, tokenizer=sentiment2_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Emotion models
emotion1_name = "j-hartmann/emotion-english-distilroberta-base"
emotion1_tokenizer = AutoTokenizer.from_pretrained(emotion1_name)
emotion1_model = AutoModelForSequenceClassification.from_pretrained(emotion1_name).to(DEVICE)
emotion1_pipe = pipeline("text-classification", model=emotion1_model, tokenizer=emotion1_tokenizer, device=0 if torch.cuda.is_available() else -1)

emotion2_name = "nateraw/bert-base-uncased-emotion"
emotion2_tokenizer = AutoTokenizer.from_pretrained(emotion2_name)
emotion2_model = AutoModelForSequenceClassification.from_pretrained(emotion2_name).to(DEVICE)
emotion2_pipe = pipeline("text-classification", model=emotion2_model, tokenizer=emotion2_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Chatbot model
chatbot_name = "microsoft/DialoGPT-medium"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_name)
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_name).to(DEVICE)

# ------------ HELPER FUNCTIONS -----------------
def get_gpu_mem():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated(0) / 1024 ** 2, 1)
    return None

def get_cpu_mem():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 ** 2, 1)

def timed_fn(fn):
    def timed(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return timed

class ViswaModel:
    def __init__(self):
        self.sentiment_pipes = [sentiment1_pipe, sentiment2_pipe]
        self.emotion_pipes = [emotion1_pipe, emotion2_pipe]
        self.chatbot_tokenizer = chatbot_tokenizer
        self.chatbot_model = chatbot_model
        self.memory = {}

    def predict_sentiment(self, message):
        results = []
        for pipe in self.sentiment_pipes:
            res = pipe(message)[0]
            label = res['label'].lower()
            if label in ['positive', 'pos']:
                label = 'positive'
            elif label in ['negative', 'neg']:
                label = 'negative'
            results.append((label, float(res['score'])))
        labels = [lbl for lbl, score in results]
        best_label = max(set(labels), key=labels.count)
        avg_conf = sum(score for lbl, score in results)/len(results)
        return best_label, avg_conf, results

    def predict_emotion(self, message):
        results = []
        for pipe in self.emotion_pipes:
            res = pipe(message)[0]
            label = res['label'].lower()
            results.append((label, float(res['score'])))
        best = max(results, key=lambda x: x[1])
        return best[0], best[1], results

    def chatbot_response(self, user_id, message):
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append(message)
        bot_input = " ".join(self.memory[user_id][-6:]) + self.chatbot_tokenizer.eos_token
        bot_input_ids = self.chatbot_tokenizer.encode(bot_input, return_tensors="pt").to(DEVICE)
        response_ids = self.chatbot_model.generate(
            bot_input_ids,
            max_length=500,
            pad_token_id=self.chatbot_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        response = self.chatbot_tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        self.memory[user_id].append(response)
        return response

viswamodel = ViswaModel()

def save_to_db(user_id, user_message, bot_response, sentiment, emotion, confidence):
    conn = sqlite3.connect("viswamodel_users.db")
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO chat_history (user_id, timestamp, user_message, bot_response, sentiment, emotion, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, timestamp, user_message, bot_response, sentiment, emotion, confidence))
    conn.commit()
    conn.close()

def mental_health_suggestion(emotion):
    suggestions = {
        "joy": "Keep enjoying your moments! Try journaling to remember happy days.",
        "sadness": "It might help to talk to a trusted friend or counselor.",
        "anger": "Consider taking a walk or deep breathing exercises.",
        "fear": "Try grounding techniques and speak to someone supportive.",
        "surprise": "Unexpected moments can be good or bad. Take a pause to process.",
        "disgust": "Try to identify what caused the feeling and avoid it if possible.",
        "neutral": "Keep engaging in what you're doing or try something relaxing."
    }
    return suggestions.get(emotion.lower(), "Take care! Try expressing your thoughts in a journal.")

# Performance wrappers
sentiment_timed = timed_fn(viswamodel.predict_sentiment)
emotion_timed = timed_fn(viswamodel.predict_emotion)
chatbot_timed = timed_fn(viswamodel.chatbot_response)
save_db_timed = timed_fn(save_to_db)

# ------------ STREAMLIT UI ---------------------
st.set_page_config(page_title="🧠 Viswamodel Mental Health Support Chatbot", page_icon="🤖")

st.title("🧠 Viswamodel Mental Health Support Chatbot")

user_id = st.text_input("Enter your user ID")
if "history" not in st.session_state:
    st.session_state.history = []
if "perf_history" not in st.session_state:
    st.session_state.perf_history = []

user_input = st.text_input("How are you feeling today?")

if st.button("Send") and user_input and user_id:
    (sentiment, sent_conf, sent_models), sent_time = sentiment_timed(user_input)
    (emotion, emo_conf, emo_models), emo_time = emotion_timed(user_input)
    bot_response, resp_time = chatbot_timed(user_id, user_input)
    _, db_time = save_db_timed(user_id, user_input, bot_response, sentiment, emotion, sent_conf)
    advice = mental_health_suggestion(emotion)
    active_users = len(viswamodel.memory)
    gpu_mem = get_gpu_mem()
    cpu_mem = get_cpu_mem()

    perf = {
        "sent_time_sec": sent_time,
        "emo_time_sec": emo_time,
        "resp_time_sec": resp_time,
        "db_time_sec": db_time,
        "gpu_mem_mb": gpu_mem,
        "cpu_mem_mb": cpu_mem,
        "active_conversations": active_users,
        "sent_models": sent_models,
        "emo_models": emo_models
    }

    st.session_state.history.append((user_input, bot_response, sentiment, emotion, advice, sent_conf, emo_conf))
    st.session_state.perf_history.append(perf)

for i, (u, b, s, e, adv, c1, c2) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**You:** {u}")
    st.markdown(f"**Bot:** {b}")
    st.markdown(f"*Sentiment (ensemble):* {s} _(conf: {c1:.2f})_")
    st.markdown(f"*Emotion (ensemble):* {e} _(conf: {c2:.2f})_")
    st.markdown(f"*Advice:* {adv}")
    st.markdown("---")

if st.session_state.perf_history:
    st.header("Performance Metrics (latest):")
    perf = st.session_state.perf_history[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment Latency (s)", f"{perf['sent_time_sec']:.3f}")
    col2.metric("Emotion Latency (s)", f"{perf['emo_time_sec']:.3f}")
    col3.metric("Response Latency (s)", f"{perf['resp_time_sec']:.3f}")
    st.write(f"DB Write Time: {perf['db_time_sec']:.3f} s")
    st.write(f"GPU Mem (MB): {perf['gpu_mem_mb']}")
    st.write(f"CPU Mem (MB): {perf['cpu_mem_mb']}")
    st.write(f"Active Conversations: {perf['active_conversations']}")
    with st.expander("Show detailed model outputs"):
        st.write("Sentiment model outputs:", perf['sent_models'])
        st.write("Emotion model outputs:", perf['emo_models'])

st.info("Model ensemble: Sentiment uses CardiffNLP + DistilBERT, Emotion uses Hartmann + BERT-emotion, Chatbot uses DialoGPT-medium.")
