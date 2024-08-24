# Fake-news-detection
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Paths
DATA_PATH = 'data/news.csv'
MODEL_PATH = 'model/fake_news_model.pkl'

# Load the dataset
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"The file {path} does not exist.")

# Preprocess the data
def preprocess_data(df):
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Save the vectorizer and model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    
    return model, vectorizer

# Evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

# Main function
def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, vectorizer = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
