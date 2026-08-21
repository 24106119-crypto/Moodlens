### MoodLens – Smart Sentiment & Emotion Analyzer

MoodLens is a lightweight, self-contained Python web application that analyzes the sentiment, emotion, and key drivers of text inputs. It operates 100% locally using standard machine learning libraries, requiring no internet connection, external API keys, or database configurations. 

### Features

* **Multi-Dimensional Analysis:** Computes sentiment classification (Positive/Negative/Neutral), confidence scores, and numerical sentiment polarity scores (-1.0 to +1.0).
* **Emotion Prediction:** Maps text inputs to emotional states including Joy, Sadness, Anger, Fear, Surprise, or Neutral.
* **Explainable Output:** Extracts key driving words and provides structured explanations for why a specific sentiment was achieved.
* **Session Tracking:** Maintains a local session history table and visualizes sentiment distribution using a dynamically updating bar chart.
* **Resilient Input Handling:** Native error boundaries for empty, malformed, or character-heavy text inputs.

### Architecture & Stack

* **UI Framework:** Streamlit
* **Data Management:** Pandas
* **Machine Learning:** Scikit-learn (TF-IDF Vectorizer + Logistic Regression)
* **Model Persistence:** Joblib
* **Visualization:** Matplotlib

### Repository Structure

text

moodlens/
├── app.py                 # Streamlit web application interface
├── train_model.py         # Model training and serialization script
├── dataset.csv            # Labeled training dataset 
├── requirements.txt       # Python package dependencies
├── sentiment_model.pkl    # Serialized Logistic Regression model (Generated)
├── tfidf_vectorizer.pkl   # Serialized TF-IDF Vectorizer matrix (Generated)
└── README.md              # Project documentation

Use code with caution.

### Installation

### Prerequisites

* Python 3.9 or higher installed on your system.

### Setup

1. Clone or navigate into the repository root directory: 

bash

cd moodlens

Use code with caution.
2. Install the required Python packages: 

bash

python -m pip install -r requirements.txt

Use code with caution.

### Model Training

The classification models must be compiled and serialized locally before running the web application. Execute the training script to process dataset.csv and generate the pipeline artifacts: 

bash

python train_model.py

Use code with caution.

Upon completion, sentiment_model.pkl and tfidf_vectorizer.pkl will be generated in the root directory. 

*Note: The core dataset serves as a baseline model benchmark. You can expand the classification vocabulary by appending labeled rows directly to dataset.csv and re-running the training script.* 

### Execution

Start the local Streamlit development server using the following command: 

bash

python -m streamlit run app.py

Use code with caution.

The application will initialize and output network access URLs. If your web browser does not launch automatically, open the local address provided in the terminal output: 

text

Local URL: http://localhost:8501

Use code with caution.

### Planned Enhancements

* Internationalization (i18n) support for multi-language text processing.
* Batch processing pipelines via bulk CSV file uploads.
* Session history persistence through CSV export utilities.
* Expanded multi-class emotion datasets to replace heuristic keyword rules.
