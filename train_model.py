"""
train_model.py
----------------
This script trains a simple Machine Learning model that can look at a
sentence and guess whether it is Positive, Negative, or Neutral.

HOW IT WORKS (in simple terms):
1. We read a CSV file (dataset.csv) that has example sentences and their
   correct sentiment label.
2. We turn the sentences into numbers using something called TF-IDF.
   (TF-IDF just measures how important each word is in a sentence.)
3. We teach a Logistic Regression model to recognize patterns using
   those numbers.
4. We save the trained model and the TF-IDF converter to disk using
   joblib, so the app.py file can load them later without retraining.

You only need to run this file ONCE (or again if you change the dataset).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# STEP 1: Load the dataset
# -------------------------
print("Step 1: Loading dataset.csv ...")
data = pd.read_csv("dataset.csv")
print(f"Loaded {len(data)} sample sentences.")
print(data["label"].value_counts())

# The text column holds the sentences, the label column holds the answer
X = data["text"]
y = data["label"]

# STEP 2: Split data into training and testing sets
# ----------------------------------------------------
# We keep some data aside (20%) to check how good the model is after training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 3: Convert text into numbers using TF-IDF
# --------------------------------------------------
# Computers cannot understand words directly, so we convert sentences
# into numeric vectors. TF-IDF gives higher importance to meaningful
# words and lower importance to common words like "the" or "is".
print("\nStep 2: Converting text to numbers with TF-IDF ...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,       # we KEEP words like "not" and "very" - they matter for sentiment!
    max_features=3000,
    ngram_range=(1, 1)     # single words only, works best for this small dataset
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# STEP 4: Train the Logistic Regression model
# -----------------------------------------------
print("\nStep 3: Training the Logistic Regression model ...")
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vectors, y_train)

# STEP 5: Check how accurate the model is
# --------------------------------------------
predictions = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel accuracy on test data: {accuracy * 100:.2f}%")
print("(Note: this uses a small demo dataset of 120 sentences, so accuracy")
print(" is just a rough guide. Adding more rows to dataset.csv and re-running")
print(" this script will improve accuracy.)")

# STEP 6: Save the trained model and vectorizer to disk
# -----------------------------------------------------------
# joblib.dump() saves Python objects into files so we can reuse them
# later inside app.py without training again every time.
print("\nStep 4: Saving model files ...")
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nAll done! Two files were created in this folder:")
print("  - sentiment_model.pkl      (the trained model)")
print("  - tfidf_vectorizer.pkl     (the text-to-number converter)")
print("\nYou can now run the app with: python -m streamlit run app.py")
