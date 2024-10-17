from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('.h5')  # Update path
with open('', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the LabelEncoder (used for converting sentiment predictions back to labels)
with open('', 'rb') as handle:
    le = pickle.load(handle)

max_sequence_length = 200  # Ensure this matches your training configuration

# Function to clean text (ensure this matches the cleaning process used in training)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()
    text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]  # Remove stop words
    text = ' '.join(text)
    return text

@app.route('/')
def home():
    return "Sentiment Analysis Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if 'review' is in the incoming JSON request
    data = request.json
    if 'review' not in data:
        return jsonify({'error': 'No review provided'}), 400

    # Process the review and make a prediction
    review = data['review']
    cleaned_review = clean_text(review)
    review_seq = tokenizer.texts_to_sequences([cleaned_review])
    review_pad = pad_sequences(review_seq, maxlen=max_sequence_length)
    prediction = model.predict(review_pad)

    # Convert prediction to sentiment label
    predicted_sentiment = le.classes_[np.argmax(prediction)]

    return jsonify({'review': review, 'predicted_sentiment': predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
