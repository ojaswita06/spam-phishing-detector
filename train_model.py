import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import re

data = pd.read_csv("emails.csv")

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

data['text'] = data['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB())
])

pipeline.fit(X_train, y_train)

with open("spam_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully!")

