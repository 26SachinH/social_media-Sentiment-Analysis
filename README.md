🧠 **Social Media Sentiment Analysis Web App
📖 Overview**

This project is a Flask-based Sentiment Analysis Web Application that uses Natural Language Processing (NLP) and Machine Learning to classify social media text into Positive, Negative, or Neutral sentiments.
It helps analyze opinions, emotions, or reactions expressed in social media posts, tweets, or reviews.

🚀 **Features**

🌐 Simple and interactive Flask web interface for text input

🤖 Machine Learning model built using TF-IDF Vectorization and Logistic Regression

📊 Classifies sentiment into Positive, Negative, or Neutral

💾 Pre-trained model saved as sentiment_model.pkl for fast predictions

🔁 Automatically retrains the model if the saved file is missing

🧹 Preprocessing pipeline for text cleaning and standardization

🧩 **Project Structure
📁 sentiment-analysis-app/**
│
├── app.py                  # Main Flask application
├── sentiment_model.pkl     # Pre-trained sentiment model
├── templates/
│   └── index.html          # Frontend template for user input
├── static/                 # (Optional) for CSS, JS, or image files
└── README.md               # Project documentation

⚙️ **How It Works**

User enters a text message or comment in the web interface.

The Flask app passes the input text to the trained model pipeline.

The model uses TF-IDF features and Logistic Regression to classify the sentiment.

The result (Positive / Negative / Neutral) is displayed as output on the webpage.

🧠 **Model Training Details**

**Algorithm:** Logistic Regression

**Vectorization:** TF-IDF (with unigrams and bigrams, max features = 5000)

**Training Data:** Social media text labeled as Positive, Negative, or Neutral

**Libraries Used:**

pandas

scikit-learn

joblib

Flask

The model is trained automatically if sentiment_model.pkl is not found in the project folder.

🖥️ **How to Run Locally**
🔧 **Step 1:** Clone the Repository
git clone https://github.com/<your-username>/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis

🔧 **Step 2:** Install Dependencies
pip install -r requirements.txt


(If you don’t have a requirements.txt yet, I can generate one for you.)

🔧 **Step 3:** Run the App
python app.py

🔧 **Step 4:** Open in Browser

**Go to** 👉 http://localhost:5001

🧾 **Example Output**

**Input:**

“I love this product, it’s amazing!”

**Output:**

**Sentiment:** Positive ✅

🛠️ **Technologies Used**

Python (Flask) – Backend Web Framework

scikit-learn – Machine Learning Model

Pandas – Data Preprocessing

Joblib – Model Persistence

Frontend- HTML/CSS (Jinja Templates) 
<img width="1918" height="906" alt="Screenshot 2025-10-27 214929" src="https://github.com/user-attachments/assets/af0ae580-7600-4b4d-80ad-93af85d5ceed" />
