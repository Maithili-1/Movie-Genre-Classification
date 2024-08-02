#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Constants
ALL_GENRES = [
    'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 
    'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music', 
    'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 
    'sport', 'talk-show', 'thriller', 'war', 'western'
]
UNKNOWN_GENRE = 'Unknown'

# Function to load data
def load_data(filepath, columns, description):
    try:
        with tqdm(total=100, desc=f"Loading {description} Data") as bar:
            df = pd.read_csv(filepath, sep=':::', header=None, names=columns, engine='python')
            bar.update(100)
        return df
    except Exception as error:
        print(f"Failed to load {description} data: {error}")
        raise

# Load training data
train_df = load_data('train_data.txt', ['ID', 'Title', 'Genre', 'Plot'], "Training")

# Preprocess training data
train_plots = train_df['Plot'].astype(str).str.lower()
genre_labels = [g.split(', ') for g in train_df['Genre']]

# MultiLabel Binarizer for genre labels
mlb = MultiLabelBinarizer()
train_genres = mlb.fit_transform(genre_labels)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Vectorize training data
with tqdm(total=100, desc="Vectorizing Training Data") as bar:
    tfidf_train = vectorizer.fit_transform(train_plots)
    bar.update(100)

# Train Logistic Regression Model
with tqdm(total=100, desc="Training Logistic Regression Model") as bar:
    lr_model = LogisticRegression(max_iter=1000)
    multi_lr = MultiOutputClassifier(lr_model)
    multi_lr.fit(tfidf_train, train_genres)
    bar.update(100)

# Load test data
test_df = load_data('test_data.txt', ['ID', 'Title', 'Plot'], "Test")

# Preprocess test data
test_plots = test_df['Plot'].astype(str).str.lower()

# Vectorize test data
with tqdm(total=100, desc="Vectorizing Test Data") as bar:
    tfidf_test = vectorizer.transform(test_plots)
    bar.update(100)

# Predict genres for test data
with tqdm(total=100, desc="Predicting Genres for Test Data") as bar:
    test_predictions = multi_lr.predict(tfidf_test)
    bar.update(100)

# Prepare prediction results
predicted_genres = mlb.inverse_transform(test_predictions)
result_df = pd.DataFrame({'Title': test_df['Title'], 'Predicted_Genres': predicted_genres})
result_df['Predicted_Genres'] = result_df['Predicted_Genres'].apply(lambda g: [UNKNOWN_GENRE] if len(g) == 0 else g)

# Save prediction results
with open("genre_predictions_lr.txt", "w", encoding="utf-8") as result_file:
    for _, row in result_df.iterrows():
        movie_title = row['Title']
        genres_str = ', '.join(row['Predicted_Genres'])
        result_file.write(f"{movie_title} ::: {genres_str}\n")

# Evaluate model performance
train_predictions = multi_lr.predict(tfidf_train)
acc = accuracy_score(train_genres, train_predictions)
prec = precision_score(train_genres, train_predictions, average='micro', zero_division=0)
rec = recall_score(train_genres, train_predictions, average='micro', zero_division=0)
f1 = f1_score(train_genres, train_predictions, average='micro', zero_division=0)

# Save evaluation metrics
with open("genre_predictions_lr.txt", "a", encoding="utf-8") as result_file:
    result_file.write("\n\nModel Performance Metrics:\n")
    result_file.write(f"Accuracy: {acc * 100:.2f}%\n")
    result_file.write(f"Precision: {prec:.2f}\n")
    result_file.write(f"Recall: {rec:.2f}\n")
    result_file.write(f"F1-Score: {f1:.2f}\n")
    result_file.write("\nDetailed Classification Report:\n")
    report = classification_report(train_genres, train_predictions, target_names=mlb.classes_, zero_division=0)
    result_file.write(report)

print("Prediction results and evaluation metrics have been saved to 'genre_predictions_lr.txt'.")


ss


