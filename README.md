# 🎭 MoodLens – Smart Sentiment & Emotion Analyzer

## 🎯 Project Objective
MoodLens is a beginner-friendly Python web app that analyzes any sentence,
review, or comment and tells you:
- How positive or negative it is
- What emotion it likely expresses
- Which words drove that result
- A plain-English explanation of why

Everything runs **100% locally on your laptop** — no internet connection,
no external AI API, no login, and no complicated server setup required.

## ✨ Features
- **Sentiment detection** — Positive, Negative, or Neutral
- **Confidence percentage** — how sure the model is about its answer
- **Sentiment score** — a single number from -1 (very negative) to +1 (very positive)
- **Mood Meter** — a visual scale from Very Negative to Very Positive
- **Emotion prediction** — Joy, Sadness, Anger, Fear, Surprise, or Neutral
- **Keyword extraction** — the most important words in your text
- **Smart Explanation** — a simple, template-based reason for the result (no AI API used)
- **Session history** — a table of everything you've analyzed so far
- **Sentiment distribution chart** — a bar chart summarizing your session
- **Sample text buttons** — try the app instantly with one click
- **Friendly error handling** — the app never crashes on empty or long input

## 🛠️ Technologies Used
- **Python 3**
- **Streamlit** – for the web interface
- **pandas** – for handling the dataset and history table
- **scikit-learn** – TF-IDF Vectorizer + Logistic Regression for sentiment classification
- **joblib** – for saving/loading the trained model
- **matplotlib** – for the sentiment distribution chart

No FastAPI, Flask, REST API, React, Node.js, or database of any kind is used.

## 📁 Folder Structure
```
moodlens/
│
├── app.py                 # The Streamlit website (run this to use the app)
├── train_model.py         # Trains the ML model (run this first)
├── dataset.csv            # 120 built-in sample sentences (40 positive, 40 negative, 40 neutral)
├── requirements.txt       # List of Python packages needed
├── sentiment_model.pkl    # Created automatically after training
├── tfidf_vectorizer.pkl   # Created automatically after training
└── README.md              # This file
```

**Where to place files:** Put all of the files above inside one folder
named `moodlens` on your computer (for example, `C:\Users\YourName\moodlens`).
Then open that folder in VS Code.

## 🚀 Installation (Windows)

1. Make sure Python 3.9 or newer is installed. Check with:
   ```
   python --version
   ```
2. Open a terminal (Command Prompt or VS Code terminal) **inside the
   `moodlens` folder**.
3. Install the required packages:
   ```
   python -m pip install -r requirements.txt
   ```
   **Expected output:** You will see pip downloading and installing
   packages like `streamlit`, `pandas`, `scikit-learn`, `joblib`, and
   `matplotlib`, ending with a message like `Successfully installed ...`.

## 🧠 Training the Model

Run this command once (from inside the `moodlens` folder):
```
python train_model.py
```

**Expected output:** You will see step-by-step messages in the terminal,
something like:
```
Step 1: Loading dataset.csv ...
Loaded 120 sample sentences.
Step 2: Converting text to numbers with TF-IDF ...
Step 3: Training the Logistic Regression model ...
Model accuracy on test data: ~50-60%
Step 4: Saving model files ...
All done! Two files were created in this folder:
  - sentiment_model.pkl
  - tfidf_vectorizer.pkl
```
After this finishes, you should see two new files appear in your
`moodlens` folder: `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.

> **Note on accuracy:** The built-in dataset only has 120 example
> sentences, which keeps the project simple and fast to train. This
> means accuracy will be moderate (roughly 50–60%) rather than
> near-perfect. You can improve it by adding more rows to
> `dataset.csv` and running `python train_model.py` again.

## ▶️ Running the Application

Start the website with:
```
python -m streamlit run app.py
```

**Expected output:** Streamlit will print a message like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```
Your default web browser should open automatically. If it doesn't,
copy the `Local URL` shown in the terminal and paste it into your browser.

## 📝 Example Inputs
Try typing (or clicking the sample buttons for) sentences like:
- `"I absolutely loved this product!"` → Positive, Joy
- `"The service was terrible."` → Negative, Sadness/Anger
- `"The package arrived today."` → Neutral, Neutral
- `"I am furious and disgusted by this awful experience."` → Negative, Anger

## 📊 Expected Outputs
After clicking **Analyze**, you should see:
1. A sentiment label with an emoji (e.g. "😊 Sentiment: Positive")
2. A confidence percentage (e.g. "72.4%")
3. A sentiment score between -1 and +1
4. A Mood Meter progress bar showing where the text falls
5. A predicted emotion with its own emoji
6. A short list of important keywords
7. A one-paragraph Smart Explanation
8. A new row added to the Session Analysis History table below
9. An updated bar chart showing your session's sentiment mix

## 🔮 Future Enhancements
- Support for multiple languages
- Ability to upload a CSV of many texts and analyze them all at once
- Export session history to a CSV file
- More advanced emotion detection using a trained ML model instead of keyword rules
- Dark mode / theme customization

---
Built with ❤️ using Python and Streamlit — runs entirely on your own laptop.
