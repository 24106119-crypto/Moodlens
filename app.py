"""
app.py
--------
This is the MoodLens website. It uses Streamlit to create a simple
webpage where you can type in a sentence and instantly see:

- The sentiment (Positive / Negative / Neutral)
- A confidence percentage
- A sentiment score from -1 to +1
- A "Mood Meter" showing where the text falls on a scale
- A predicted emotion (Joy, Sadness, Anger, Fear, Surprise, Neutral)
- The most important keywords in the text
- A simple plain-English explanation
- A history of everything you analyzed in this session
- A small chart showing the mix of sentiments so far

To run this file, you must first train the model by running:
    python train_model.py

Then start the website with:
    python -m streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="MoodLens", page_icon="🎭", layout="centered")

st.title("🎭 MoodLens")
st.subheader("Smart Sentiment & Emotion Analyzer")
st.write("Type any sentence, review, or comment below and let MoodLens analyze it.")

# ------------------------------------------------------------------
# LOAD THE TRAINED MODEL AND VECTORIZER
# ------------------------------------------------------------------
# We check if the model files exist. If they don't, we stop and tell
# the user to run train_model.py first, instead of crashing.
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error(
        "⚠️ Model files not found!\n\n"
        "Please train the model first by running this command in your terminal:\n\n"
        "`python train_model.py`\n\n"
        "Then restart this app with `python -m streamlit run app.py`."
    )
    st.stop()  # This safely stops the app from running further

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ------------------------------------------------------------------
# SIMPLE EMOTION KEYWORD LISTS (rule-based, no external AI needed)
# ------------------------------------------------------------------
# We look for words related to each emotion inside the user's text.
# This is a simple and beginner-friendly way to guess the emotion.
EMOTION_KEYWORDS = {
    "Joy": ["happy", "love", "loved", "great", "amazing", "wonderful", "joy",
            "excited", "delighted", "fantastic", "glad", "thrilled", "perfect",
            "awesome", "beautiful", "grateful", "proud", "fun", "enjoy"],
    "Sadness": ["sad", "unhappy", "disappointed", "regret", "cry", "down",
                "depressed", "upset", "heartbroken", "miserable", "lonely",
                "hopeless", "gloomy", "let down", "tired", "exhausted"],
    "Anger": ["angry", "furious", "hate", "annoyed", "mad", "rude", "unacceptable",
              "outrageous", "irritated", "frustrated", "disgusted", "terrible"],
    "Fear": ["scared", "afraid", "worried", "nervous", "anxious", "terrified",
             "fear", "panic", "concerned", "uneasy"],
    "Surprise": ["surprised", "shocked", "unexpected", "wow", "unbelievable",
                 "sudden", "amazed", "astonished", "surprise"],
}

# Words used by the Smart Explanation feature
EXPLANATION_WORDS = {
    "positive": "satisfaction, happiness, or approval",
    "negative": "frustration, disappointment, or dissatisfaction",
    "neutral": "plain facts without strong emotional opinion",
}

# ------------------------------------------------------------------
# SESSION HISTORY (keeps track of everything analyzed this session)
# ------------------------------------------------------------------
# st.session_state lets us remember data while the app is open,
# without needing a database.
if "history" not in st.session_state:
    st.session_state.history = []  # will hold dictionaries of past results


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def predict_sentiment(text):
    """
    Takes raw text, converts it to numbers using the TF-IDF vectorizer,
    and returns the predicted label plus the probability for each class.
    """
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    # Build a simple dictionary like {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
    prob_dict = dict(zip(model.classes_, probabilities))
    return prediction, prob_dict


def compute_sentiment_score(prob_dict):
    """
    Turns the probabilities into a single score from -1 (very negative)
    to +1 (very positive). Positive probability pushes the score up,
    negative probability pushes it down, and neutral has no effect.
    """
    pos = prob_dict.get("positive", 0)
    neg = prob_dict.get("negative", 0)
    score = pos - neg
    return round(score, 2)


def get_mood_label(score):
    """
    Converts the numeric score into a friendly Mood Meter label.
    """
    if score <= -0.6:
        return "Very Negative", 0
    elif score <= -0.2:
        return "Negative", 1
    elif score < 0.2:
        return "Neutral", 2
    elif score < 0.6:
        return "Positive", 3
    else:
        return "Very Positive", 4


def detect_emotion(text, sentiment):
    """
    Looks for emotion keywords inside the text. If none are found,
    it falls back to a sensible default based on the overall sentiment.
    """
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for word in keywords if word in text_lower)
        if count > 0:
            scores[emotion] = count

    if scores:
        # Pick the emotion with the most keyword matches
        best_emotion = max(scores, key=scores.get)
        return best_emotion

    # Fallback: no emotion keywords matched, so guess from sentiment
    if sentiment == "positive":
        return "Joy"
    elif sentiment == "negative":
        return "Sadness"
    else:
        return "Neutral"


def extract_keywords(text, top_n=5):
    """
    Uses the same TF-IDF vectorizer to find which words in the text
    were considered most important by the model.
    """
    text_vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF score for each word in this specific text
    scores = text_vector.toarray()[0]

    # Pair each word with its score, then keep only words that appear (score > 0)
    word_scores = [(feature_names[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]

    # Sort by score, highest first, and take the top ones
    word_scores.sort(key=lambda pair: pair[1], reverse=True)
    top_words = [word for word, score in word_scores[:top_n]]

    return top_words if top_words else ["(no strong keywords found)"]


def generate_explanation(sentiment, keywords):
    """
    Builds a simple, template-based explanation. No external AI is used,
    just plain Python string formatting.
    """
    reason = EXPLANATION_WORDS.get(sentiment, "a mix of tones")

    if keywords and keywords[0] != "(no strong keywords found)":
        keyword_text = ", ".join(keywords[:3])
        return (
            f"This text appears **{sentiment}** because it contains words "
            f"such as *{keyword_text}*, which are often associated with {reason}."
        )
    else:
        return (
            f"This text appears **{sentiment}** based on its overall tone, "
            f"which is often associated with {reason}."
        )


def get_emoji(sentiment):
    """Returns a simple emoji for a given sentiment label."""
    return {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "🤔")


# ------------------------------------------------------------------
# SAMPLE TEXT BUTTONS
# ------------------------------------------------------------------
st.write("**Try a sample:**")
sample_col1, sample_col2, sample_col3 = st.columns(3)

# We store the chosen sample text in session_state so the text box can use it
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

with sample_col1:
    if st.button("😊 Positive example"):
        st.session_state.input_text = "I absolutely loved this product!"
with sample_col2:
    if st.button("😞 Negative example"):
        st.session_state.input_text = "The service was terrible."
with sample_col3:
    if st.button("😐 Neutral example"):
        st.session_state.input_text = "The package arrived today."

# ------------------------------------------------------------------
# TEXT INPUT AREA
# ------------------------------------------------------------------
user_text = st.text_area(
    "Enter your text here:",
    value=st.session_state.input_text,
    height=120,
    max_chars=2000,  # keeps very long text from causing problems
    placeholder="Example: The new update made everything so much faster!"
)

analyze_clicked = st.button("🔍 Analyze", type="primary")

# ------------------------------------------------------------------
# MAIN ANALYSIS LOGIC
# ------------------------------------------------------------------
if analyze_clicked:
    # ERROR HANDLING: empty text
    if not user_text or not user_text.strip():
        st.warning("⚠️ Please enter some text before clicking Analyze.")
    else:
        # Wrapping in try/except so the app never crashes on unexpected input
        try:
            clean_text = user_text.strip()

            sentiment, prob_dict = predict_sentiment(clean_text)
            confidence = round(max(prob_dict.values()) * 100, 1)
            score = compute_sentiment_score(prob_dict)
            mood_label, mood_position = get_mood_label(score)
            emotion = detect_emotion(clean_text, sentiment)
            keywords = extract_keywords(clean_text)
            explanation = generate_explanation(sentiment, keywords)
            emoji = get_emoji(sentiment)

            # ---- RESULT CARD ----
            st.markdown("---")
            st.markdown(f"### {emoji} Sentiment: **{sentiment.capitalize()}**")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric("Confidence", f"{confidence}%")
            with result_col2:
                st.metric("Sentiment Score", f"{score}")

            # ---- MOOD METER ----
            st.markdown("#### 🎚️ Mood Meter")
            mood_scale = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            st.progress(mood_position / (len(mood_scale) - 1))
            st.write(" | ".join(
                f"**[{m}]**" if m == mood_label else m for m in mood_scale
            ))

            # ---- EMOTION ----
            emotion_emojis = {
                "Joy": "😄", "Sadness": "😢", "Anger": "😠",
                "Fear": "😨", "Surprise": "😲", "Neutral": "😐"
            }
            st.markdown(f"#### 🎭 Predicted Emotion: {emotion_emojis.get(emotion, '')} **{emotion}**")

            # ---- KEYWORDS ----
            st.markdown("#### 🔑 Important Keywords")
            st.write(", ".join(keywords))

            # ---- EXPLANATION ----
            st.markdown("#### 💡 Smart Explanation")
            st.info(explanation)

            # ---- SAVE TO SESSION HISTORY ----
            st.session_state.history.append({
                "Text": clean_text if len(clean_text) < 60 else clean_text[:57] + "...",
                "Sentiment": sentiment.capitalize(),
                "Emotion": emotion,
                "Confidence": f"{confidence}%",
                "Score": score,
            })

        except Exception as error:
            # This is a safety net so the app never crashes.
            st.error(f"Something went wrong while analyzing your text: {error}")

# ------------------------------------------------------------------
# SESSION HISTORY TABLE
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🕘 Session Analysis History")

if len(st.session_state.history) == 0:
    st.write("No analysis yet. Enter some text above and click Analyze!")
else:
    history_df = pd.DataFrame(st.session_state.history)
    # Show the most recent entries first
    st.dataframe(history_df.iloc[::-1], use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    # ------------------------------------------------------------------
    # SENTIMENT DISTRIBUTION CHART
    # ------------------------------------------------------------------
    st.markdown("### 📊 Sentiment Distribution (this session)")
    counts = history_df["Sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
    bar_colors = [colors.get(label, "#2196F3") for label in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors)
    ax.set_ylabel("Number of texts")
    ax.set_title("Sentiment Counts")
    st.pyplot(fig)

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.caption("MoodLens — a simple, local, beginner-friendly sentiment & emotion analyzer.")
