import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
import os

# ---------- CONFIG ----------
DATA_PATH = "social_network_spam.csv"
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ---------- CLEAN TEXT FUNCTION ----------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)     # remove mentions
    text = re.sub(r'#\w+', '', text)     # remove hashtags
    text = re.sub(r'\W', ' ', text)      # remove special characters
    return text.lower()

# ---------- LOAD AND PREPROCESS ----------
def preprocess_and_split():
    print("🔄 Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- TRAIN MODEL ----------
def train_model(X_train, X_test, y_train, y_test):
    print("🏋️ Training model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\n✅ Training Complete\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("💾 Model and vectorizer saved.\n")

# ---------- PREDICT FUNCTION ----------
def predict_spam(text):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# ---------- FLASK APP ----------
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Welcome to the Spam & Fake User Detection API!"

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text field is required."}), 400
    result = predict_spam(text)
    return jsonify({"prediction": result})

# ---------- MAIN RUN ----------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        X_train, X_test, y_train, y_test = preprocess_and_split()
        train_model(X_train, X_test, y_train, y_test)
    else:
        print("✅ Model already trained. Ready to serve predictions.\n")

    # Run prediction test
    test_text = "Congratulations! You have won a free iPhone. Click here!"
    print(f"📨 Sample prediction for: \"{test_text}\"")
    print("🔮 Prediction:", predict_spam(test_text), "\n")

    # Start the Flask server
    print("🚀 Starting Flask API at http://127.0.0.1:5000")
    app.run(debug=True)
