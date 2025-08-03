# ------------------------------------------------
# SECTION 1: Dependencies and Environment Setup
# ------------------------------------------------
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM
)
import random
import json
from flask import Flask, request, jsonify
import sqlite3
import os
from datetime import datetime
import logging
import streamlit as st
import time
import psutil  # new

# Logging setup
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Ensure GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# SECTION 2: Database Setup
# -------------------------------
def init_db():
    conn = sqlite3.connect("chatbot_users.db")
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

# -------------------------------
# SECTION 3: Sentiment & Emotion Analysis Setup
# -------------------------------
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(DEVICE)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0 if torch.cuda.is_available() else -1)

emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(DEVICE)
emotion_analyzer = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, device=0 if torch.cuda.is_available() else -1)

# -------------------------------
# SECTION 4: Context-Aware Chatbot Setup (DialoGPT)
# -------------------------------
chatbot_model_name = "microsoft/DialoGPT-medium"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_model_name).to(DEVICE)
conversation_memory = {}

# -------------------------------
# SECTION 5: PERFORMANCE METRICS HELPERS
# -------------------------------
def get_gpu_mem():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated(0) / 1024 ** 2, 1)  # in MB
    return None

def get_cpu_mem():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 ** 2, 1)  # in MB

def timed_fn(fn):
    def timed(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return timed

# --- Wrap core functions with timing ---
def analyze_sentiment(message):
    result = sentiment_analyzer(message)[0]
    return result['label'], float(result['score'])

def analyze_emotion(message):
    result = emotion_analyzer(message)[0]
    return result['label']

def generate_response(user_id, user_input):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append(user_input)

    bot_input = " ".join(conversation_memory[user_id][-6:]) + chatbot_tokenizer.eos_token
    bot_input_ids = chatbot_tokenizer.encode(bot_input, return_tensors='pt').to(DEVICE)

    chat_history_ids = chatbot_model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=chatbot_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    conversation_memory[user_id].append(response)
    return response

def save_to_db(user_id, user_message, bot_response, sentiment, emotion, confidence):
    conn = sqlite3.connect("chatbot_users.db")
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO chat_history (user_id, timestamp, user_message, bot_response, sentiment, emotion, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, timestamp, user_message, bot_response, sentiment, emotion, confidence))
    conn.commit()
    conn.close()

# --- Timed versions for perf monitoring ---
analyze_sentiment_timed = timed_fn(analyze_sentiment)
analyze_emotion_timed = timed_fn(analyze_emotion)
generate_response_timed = timed_fn(generate_response)
save_to_db_timed = timed_fn(save_to_db)
# ------------------------------------------

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

# -------------------------------
# SECTION 6: Streamlit UI
# -------------------------------
def launch_ui():
    st.title("🧠 Mental Health Support Chatbot")
    user_id = st.text_input("Enter your user ID")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "perf_history" not in st.session_state:
        st.session_state.perf_history = []

    user_input = st.text_input("How are you feeling today?")
    if st.button("Send") and user_input and user_id:
        (sentiment, confidence), sent_time = analyze_sentiment_timed(user_input)
        emotion, emo_time = analyze_emotion_timed(user_input)
        bot_response, resp_time = generate_response_timed(user_id, user_input)
        _, db_time = save_to_db_timed(user_id, user_input, bot_response, sentiment, emotion, confidence)
        advice = mental_health_suggestion(emotion)

        active_users = len(conversation_memory)
        gpu_mem = get_gpu_mem()
        cpu_mem = get_cpu_mem()

        perf = {
            "sent_time_sec": sent_time,
            "emo_time_sec": emo_time,
            "resp_time_sec": resp_time,
            "db_time_sec": db_time,
            "gpu_mem_mb": gpu_mem,
            "cpu_mem_mb": cpu_mem,
            "active_conversations": active_users
        }

        st.session_state.history.append((user_input, bot_response, sentiment, emotion, advice))
        st.session_state.perf_history.append(perf)

    for i, (u, b, s, e, adv) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**You:** {u}")
        st.markdown(f"**Bot:** {b}")
        st.markdown(f"*Sentiment:* {s}, *Emotion:* {e}")
        st.markdown(f"*Advice:* {adv}")
        st.markdown("---")

    if st.session_state.perf_history:
        st.header("Performance Metrics (latest):")
        perf = st.session_state.perf_history[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment Latency (s)", f"{perf['sent_time_sec']:.3f}")
        col2.metric("Emotion Latency (s)", f"{perf['emo_time_sec']:.3f}")
        col3.metric("Response Latency (s)", f"{perf['resp_time_sec']:.3f}")
        st.write(f"DB Save Time: {perf['db_time_sec']:.3f} s")
        st.write(f"GPU Mem (MB): {perf['gpu_mem_mb']}")
        st.write(f"CPU Mem (MB): {perf['cpu_mem_mb']}")
        st.write(f"Active Conversations: {perf['active_conversations']}")

# -------------------------------
# SECTION 7: Main
# -------------------------------
if __name__ == "__main__":
    launch_ui()
else:
    app = Flask(__name__)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        user_id = data.get("user_id")
        message = data.get("message")

        # Performance: measure inference latencies
        (sentiment, confidence), sent_time = analyze_sentiment_timed(message)
        emotion, emo_time = analyze_emotion_timed(message)
        bot_response, resp_time = generate_response_timed(user_id, message)
        _, db_time = save_to_db_timed(user_id, message, bot_response, sentiment, emotion, confidence)

        gpu_mem = get_gpu_mem()
        cpu_mem = get_cpu_mem()
        active_users = len(conversation_memory)

        # Log performance for monitoring
        logging.info(f"PERF: sent_time={sent_time:.3f}s, emo_time={emo_time:.3f}s, resp_time={resp_time:.3f}s, db_time={db_time:.3f}s, gpu_mem={gpu_mem} MB, cpu_mem={cpu_mem} MB, active_users={active_users}")

        return jsonify({
            "response": bot_response,
            "sentiment": sentiment,
            "emotion": emotion,
            "confidence": confidence,
            "perf": {
                "sent_time_sec": sent_time,
                "emo_time_sec": emo_time,
                "resp_time_sec": resp_time,
                "db_time_sec": db_time,
                "gpu_mem_mb": gpu_mem,
                "cpu_mem_mb": cpu_mem,
                "active_conversations": active_users
            }
        })

    @app.route("/reset", methods=["POST"])
    def reset_conversation():
        data = request.get_json()
        user_id = data.get("user_id")
        conversation_memory[user_id] = []
        return jsonify({"status": "Conversation reset."})

    @app.route("/history", methods=["GET"])
    def get_history():
        user_id = request.args.get("user_id")
        conn = sqlite3.connect("chatbot_users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user_message, bot_response, sentiment, emotion, confidence FROM chat_history WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return jsonify({"history": rows})

    app.run(debug=True, use_reloader=False)
