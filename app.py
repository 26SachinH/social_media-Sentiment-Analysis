from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create a Flask app
app = Flask(__name__)

# Function to train the model and save it
def train_model():
    file_path = (r"C:\Users\SACHIN HEBBALAKAR\OneDrive\Desktop\python\social media\social media sentamental.csv")  # Update with correct path
    data = pd.read_csv(file_path)

    # Preprocess the data
    data = data[['Text', 'Sentiment']]
    data['Sentiment'] = data['Sentiment'].str.strip()
    data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

    # Train the model
    X = data['Text']
    y = data['Sentiment']
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    model_pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(model_pipeline, 'sentiment_model.pkl')

# Check if the model is already trained and saved
try:
    model_pipeline = joblib.load('sentiment_model.pkl')  # Load pre-trained model
except FileNotFoundError:
    # If the model is not found, train it
    print("Model not found. Training a new model...")
    train_model()
    model_pipeline = joblib.load('sentiment_model.pkl')  # Load the newly trained model

# Define routes
@app.route('/')
def home():
    return render_template('index.html')  # Ensure this template exists in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')  # Get input text from form
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    # Predict sentiment
    predicted_sentiment = model_pipeline.predict([input_text])[0]
    return jsonify({'text': input_text, 'predicted_sentiment': predicted_sentiment})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
