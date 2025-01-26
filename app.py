from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
from bs4 import BeautifulSoup
import spacy
from langdetect import detect
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import time
import telebot
import openai
import re
import nltk
from nltk.stem import WordNetLemmatizer
import logging
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Token-ul de acces Hugging Face (înlocuiește cu token-ul tău)
HUGGINGFACE_TOKEN = "API"
OPENAI_KEY = "API"  # Înlocuiește cu cheia ta OpenAI
TELEGRAM_TOKEN = "API"  # Înlocuiește cu token-ul tău Telegram
ABUSEIPDB_KEY = "API"  # Înlocuiește cu cheia ta AbuseIPDB
VIRUSTOTAL_KEY = "API"  # Înlocuiește cu cheia ta VirusTotal
GOOGLE_NEWS_API_KEY = "API"  # Înlocuiește cu cheia ta Google News API

# Încarcă modelul și tokenizer-ul pentru fake news
MODEL_NAME = "dimonsleep/fake_news_detect"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)

# Încarcă modelul pentru detectarea propagandei
propaganda_model_name = "dimonsleep/antipropaganda"
propaganda_tokenizer = AutoTokenizer.from_pretrained(propaganda_model_name, token=HUGGINGFACE_TOKEN)
propaganda_model = AutoModelForSequenceClassification.from_pretrained(propaganda_model_name, token=HUGGINGFACE_TOKEN)

# Încarcă modelul pentru recunoașterea emoțiilor
emotion_model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
emotion_classifier = pipeline("text-classification", model=emotion_model_name)

# Încarcă modelul spaCy pentru extragerea de narative
nlp = spacy.load("en_core_web_sm")

# Dicționar pentru traducerea emoțiilor
emotion_translations = {
    "en": {
        "admiration": "admiration", "amusement": "amusement", "anger": "anger",
        "annoyance": "annoyance", "approval": "approval", "caring": "caring",
        "confusion": "confusion", "curiosity": "curiosity", "desire": "desire",
        "disappointment": "disappointment", "disapproval": "disapproval",
        "disgust": "disgust", "embarrassment": "embarrassment", 
        "excitement": "excitement", "fear": "fear", "gratitude": "gratitude",
        "grief": "grief", "joy": "joy", "love": "love", 
        "nervousness": "nervousness", "optimism": "optimism", 
        "pride": "pride", "realization": "realization", 
        "relief": "relief", "remorse": "remorse", 
        "sadness": "sadness", "surprise": "surprise", 
        "neutral": "neutral"
    },
    "ro": {
        "admiration": "admirație", "amusement": "amuzament", "anger": "furie",
        "annoyance": "iritație", "approval": "aprobare", "caring": "grijă",
        "confusion": "confuzie", "curiosity": "curiozitate", "desire": "dorință",
        "disappointment": "dezamăgire", "disapproval": "dezaprobare",
        "disgust": "dezgust", "embarrassment": "jenă", 
        "excitement": "entuziasm", "fear": "frică", "gratitude": "recunoștință",
        "grief": "durere", "joy": "bucurie", "love": "dragoste", 
        "nervousness": "nervozitate", "optimism": "optimism", 
        "pride": "mândrie", "realization": "realizare", 
        "relief": "ușurare", "remorse": "remușcare", 
        "sadness": "tristețe", "surprise": "surpriză", 
        "neutral": "neutru"
    }
}

def extract_text_from_url(url):
    try:
        firefox_options = Options()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--disable-gpu")
        firefox_options.add_argument("--no-sandbox")

        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)

        driver.set_page_load_timeout(30)
        driver.get(url)
        driver.implicitly_wait(10)

        text = driver.find_element(By.TAG_NAME, 'body').text
        driver.quit()

        return text if text.strip() else None

    except Exception as e:
        print(f"Eroare la extragerea textului din URL: {e}")
        return None

def predict_fake_news(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    return {
        'label': 'Fake News' if predicted_class == 0 else 'True News',
        'confidence': float(confidence)
    }

def detect_propaganda(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    propaganda_model.to(device)
    inputs = propaganda_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    propaganda_model.eval()
    with torch.no_grad():
        outputs = propaganda_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    return {
        'label': 'Propaganda' if predicted_class == 1 else 'Non-Propaganda',
        'confidence': float(confidence)
    }

def detect_emotions(text, language="en", max_length=512):
    try:
        # Use the tokenizer to truncate the text
        inputs = emotion_classifier.tokenizer(
            text, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        truncated_text = emotion_classifier.tokenizer.decode(inputs['input_ids'][0])
        
        results = emotion_classifier(truncated_text)
        emotions = {}
        for result in results:
            label = result["label"]
            score = result["score"]
            if label in emotions:
                emotions[label] += score
            else:
                emotions[label] = score
        
        total_score = sum(emotions.values())
        for label in emotions:
            emotions[label] = (emotions[label] / total_score) * 100
        
        translated_emotions = {}
        for label, score in emotions.items():
            translated_label = emotion_translations.get(language, {}).get(label, label)
            translated_emotions[translated_label] = score
        return translated_emotions
    except Exception as e:
        logging.error(f"Error in detect_emotions: {e}")
        return {}

def extract_narrative(text):
    doc = nlp(text)
    narrative = []

    for sent in doc.sents:
        entities = []
        for ent in sent.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                entities.append({'text': ent.text, 'label': ent.label_})
        if entities:
            narrative.append({
                'sentence': sent.text,
                'entities': entities
            })

    return narrative

def preprocesare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()])

def detect_manipulation_techniques(text):
    techniques = {
        "limbaj_emotional_excesiv": {
            "keywords": ["îngrozitor", "catastrofă", "teribil", "panică", "frică", "dezastru", "tragic"],
            "count": 0,
            "fragments": []
        },
        "generalizari_excesive": {
            "keywords": ["toți", "niciodată", "mereu", "nimeni", "oricând", "fiecare", "absolut"],
            "count": 0,
            "fragments": []
        },
        "atac_la_persoana": {
            "keywords": ["mincinos", "incompetent", "corupt", "lacom", "ipocrit", "inept", "manipulator"],
            "count": 0,
            "fragments": []
        }
    }

    for technique, data in techniques.items():
        for keyword in data["keywords"]:
            matches = re.finditer(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
            for match in matches:
                data["count"] += 1
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                fragment = text[start:end].strip()
                data["fragments"].append(fragment)

    return techniques

def compare_with_trusted_sources(text):
    keywords = " ".join(preprocesare_text(text).split()[:10])
    articles = get_news_articles(keywords)

    if not articles:
        return "Nu s-au găsit articole similare pe surse de încredere."

    article_texts = [article["title"] + " " + article["description"] for article in articles]
    article_texts.append(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(article_texts)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    most_similar_index = np.argmax(similarities)
    most_similar_article = articles[most_similar_index]
    similarity_score = similarities[0][most_similar_index]

    report = f"""
📰 **Comparare cu surse de încredere:**
- Articol similar găsit pe {most_similar_article['source']['name']}: "{most_similar_article['title']}".
- Similaritate: {similarity_score * 100:.2f}%
- Link: {most_similar_article['url']}
"""
    if similarity_score > 0.5:
        report += "- Concluzie: Informația este confirmată de surse de încredere."
    else:
        report += "- Concluzie: Informația nu este confirmată de surse de încredere."

    return report

def get_news_articles(query):
    logging.info(f"Interogare Google News API pentru: {query}")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": GOOGLE_NEWS_API_KEY,
        "language": "ro",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        logging.info(f"Răspuns API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "articles" in data:
                logging.info(f"Articole găsite: {len(data['articles'])}")
                return data["articles"]
            else:
                logging.warning("Nu s-au găsit articole în răspuns.")
                return []
        else:
            logging.error(f"Eroare API: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Eroare la interogarea Google News API: {e}")
        return []

def analyze_content(text):
    try:
        articol_proc = preprocesare_text(text)
        
        fake_news_result = predict_fake_news(text)
        probabilitate_falsa = fake_news_result['confidence'] * 100 if fake_news_result['label'] == 'Fake News' else 100 - fake_news_result['confidence'] * 100
        probabilitate_adevarata = 100 - probabilitate_falsa
        
        propaganda_result = detect_propaganda(text)
        probabilitate_propaganda = propaganda_result['confidence'] * 100 if propaganda_result['label'] == 'Propaganda' else 100 - propaganda_result['confidence'] * 100
        
        manipulation_techniques = detect_manipulation_techniques(text)
        manipulation_report = "🚩 **Tehnici de manipulare detectate:**\n"
        for technique, data in manipulation_techniques.items():
            if data["count"] > 0:
                manipulation_report += (
                    f"- {technique.replace('_', ' ').capitalize()}: {data['count']} fragmente identificate.\n"
                )
                for i, fragment in enumerate(data["fragments"], 1):
                    manipulation_report += f"  Fragment {i}: \"{fragment}\"\n"
            else:
                manipulation_report += f"- {technique.replace('_', ' ').capitalize()}: Nu s-au identificat fragmente.\n"
        
        emotions = detect_emotions(text)
        emotion_report = "😊 **Emoții detectate:**\n"
        for emotion, score in emotions.items():
            emotion_report += f"- {emotion}: {score:.2f}%\n"
        
        narrative = extract_narrative(text)
        narrative_report = "📖 **Narative detectate:**\n"
        for item in narrative:
            narrative_report += f"- {item['sentence']}\n"
            for entity in item['entities']:
                narrative_report += f"  - {entity['text']} ({entity['label']})\n"
        
        comparison_report = compare_with_trusted_sources(text)
        
        return f"""
🔍 **Analiza textului:**
- Probabilitate Adevărată: {probabilitate_adevarata:.2f}%
- Probabilitate Falsă: {probabilitate_falsa:.2f}%
- Predicție Știri False: {fake_news_result['label']}
- Probabilitate Propagandă: {probabilitate_propaganda:.2f}%
- Predicție Propagandă: {propaganda_result['label']}

{manipulation_report}

{emotion_report}

{narrative_report}

{comparison_report}
"""
    except Exception as e:
        logging.error(f"Eroare la analiză: {e}")
        return f"⚠️ Eroare la analiză: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Verifică dacă textul este un URL
        if text.startswith('http://') or text.startswith('https://'):
            extracted_text = extract_text_from_url(text)
            if not extracted_text:
                return jsonify({'error': 'Could not extract text from URL'}), 400
            text = extracted_text
        
        # Detectează limba textului
        try:
            language = detect(text)
        except:
            language = "en"

        # Predicții
        fake_news_result = predict_fake_news(text)
        propaganda_result = detect_propaganda(text)
        
        # Add fallback for emotions
        try:
            emotions = detect_emotions(text, language)
        except Exception as e:
            logging.error(f"Emotion detection failed: {e}")
            emotions = {}
        
        narrative = extract_narrative(text)
        manipulation_techniques = detect_manipulation_techniques(text)

        return jsonify({
            'fake_news_prediction': fake_news_result,
            'propaganda_prediction': propaganda_result,
            'narrative': narrative,
            'emotions': emotions,
            'language': language,
            'manipulation_techniques': manipulation_techniques
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
# Configurare Telegram Bot
bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🕵️ Bot de detectare fake news. Trimite-mi un text pentru verificare!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    analysis = analyze_content(text)
    bot.reply_to(message, analysis)

# Funcția pentru a rula Flask și Telegram Bot în paralel
def run_flask():
    app.run(debug=True, use_reloader=False)

def run_telegram_bot():
    bot.polling(none_stop=True)

if __name__ == '__main__':
    # Pornim Flask și Telegram Bot în thread-uri separate
    flask_thread = threading.Thread(target=run_flask)
    telegram_thread = threading.Thread(target=run_telegram_bot)

    flask_thread.start()
    telegram_thread.start()

    flask_thread.join()
    telegram_thread.join()
