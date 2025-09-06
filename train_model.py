import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import re

# Load dataset
data = pd.read_csv("emails.csv")

# Basic preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\n', ' ', text)  # remove newlines
    text = re.sub(r'\[.*?\]', '', text)  # remove [text] patterns
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# Build pipeline (vectorizer + model)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully!")
