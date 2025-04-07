# **Sentiment Analysis for Food Reviews using Yelp® Review Dataset**

## **Project Overview**
This project applies **Natural Language Processing (NLP)** techniques along with **Machine Learning (ML) and Deep Learning (DL)** models to analyze sentiment in **Yelp restaurant reviews**. The objective is to classify reviews into **Positive, Neutral, or Negative** sentiments.

The project involves:
- **Preprocessing raw text data** using NLP techniques.
- **Feature extraction** using Bag of Words (BoW) and TF-IDF.
- **Training ML models** like Naïve Bayes and Logistic Regression.
- **Training DL models** like RNN and LSTM for advanced text understanding.
- **Comparing model performance** and selecting the best models for sentiment classification.

---

## **Dataset Information**
The dataset consists of **Yelp restaurant reviews** containing:
- **Review Text** (`text`)
- **Star Ratings** (`stars`)
- **Additional Metadata** (useful votes, funny votes, timestamps)

---

## **NLP Preprocessing Steps**

### **1️⃣ Text Cleaning**
- **HTML Tag Removal:** Removes unwanted `<html>` tags using BeautifulSoup.
- **Special Characters & Punctuation Removal:** Keeps only alphanumeric text.
- **Lowercasing:** Converts all text to lowercase for uniformity.

### **2️⃣ Tokenization**
- Breaks the text into individual words (**tokens**) for further processing.

### **3️⃣ Stopword Removal**
- Common stopwords (e.g., *the, is, at, which*) are removed to retain only meaningful words.
- **Customized Stopwords List:** Important sentiment-related words like **"not", "never", "bad", "good"** are retained.

### **4️⃣ Lemmatization**
- Converts words to their base form (e.g., *running → run, better → good*).
- Ensures consistency in word representation.

### **5️⃣ Feature Engineering**
To convert text into a numerical format for model training:
- **Bag of Words (BoW):** Counts word occurrences.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Assigns importance to words based on frequency.

### **6️⃣ Word Embeddings (Deep Learning)**
- **Tokenized sequences are padded** to ensure consistent input size for LSTM and RNN models.
- **Embedding layers are used** in deep learning models to capture contextual meaning.

---

## **Models Used**
This project compares **Machine Learning** and **Deep Learning** models for sentiment classification.

### **Machine Learning Models**
- **Naïve Bayes (BoW & TF-IDF)** – Baseline probabilistic model.
- **Logistic Regression (BoW & TF-IDF)** – Efficient and interpretable model.

### **Deep Learning Models**
- **Recurrent Neural Network (RNN)** – Captures sequential relationships in text.
- **Long Short-Term Memory (LSTM)** – Improves context retention in longer texts.

---

## **Model Performance Comparison**

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------|-------------|--------------|-----------|------------|
| **Naïve Bayes (BoW)** | 81.49% | 85.33% | 81.49% | 83.05% |
| **Naïve Bayes (TF-IDF)** | 85.05% | 81.84% | 85.05% | 82.36% |
| **Logistic Regression (BoW)** | 89.67% | 88.11% | 89.67% | 88.54% |
| **Logistic Regression (TF-IDF)** | **89.90%** | **88.43%** | **89.90%** | **88.77%** |
| **RNN** | 77.82% | 75.18% | 77.82% | 75.67% |
| **LSTM** | **90.27%** | **88.68%** | **90.27%** | **88.97%** |

### **Best Performing Models**
1. **LSTM** – Best for deep learning-based sentiment classification.
2. **Logistic Regression (TF-IDF)** – Best traditional ML model.
3. **Logistic Regression (BoW)** – Efficient and competitive baseline.

---

## 📌 LSTM Model Conclusion

The LSTM model demonstrated **strong and consistent sentiment classification capabilities** across a diverse set of restaurant reviews. It accurately identified **positive**, **neutral**, and **negative** sentiments, showcasing its ability to:

- Recognize strong expressions of satisfaction (e.g., *"fantastic meal"*, *"heavenly desserts"*, *"great portion sizes"*) as **positive reviews**.
- Detect dissatisfaction and critical language (e.g., *"overpriced"*, *"stale"*, *"unprofessional service"*, *"hair in food"*) as **negative reviews**.
- Appropriately classify ambiguous or mixed-sentiment feedback (e.g., *"nothing special"*, *"a very average meal"*, *"might give it another try"*) as **neutral reviews**.

### ✅ Key Strengths

- **High accuracy and F1-score (~90%)** confirm the model’s reliability in real-world usage.
- Captures **long-range dependencies** in text, giving it an edge over traditional ML models.
- Maintains balance across all sentiment classes without overfitting to positive or negative extremes.

### 🧠 Takeaway

The LSTM model is well-suited for nuanced sentiment analysis in natural language, particularly for domains like **restaurant reviews** where emotional expression varies widely. It will serve as the **primary model** for future sentiment prediction tasks due to its robust performance and interpretability.

---

## **How to Use the LSTM Model**

```python
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Load the pretrained LSTM model
from tensorflow.keras.models import load_model
lstm_model = load_model("lstm_model_new.h5")

# Step 3: Tokenize and preprocess new review
from tensorflow.keras.preprocessing.sequence import pad_sequences

sample_review = ["The food was terrible and overpriced."]
sample_seq = tokenizer.texts_to_sequences(sample_review)
sample_pad = pad_sequences(sample_seq, maxlen=100)

# Step 4: Predict sentiment
import numpy as np
prediction = np.argmax(lstm_model.predict(sample_pad), axis=1)
print("LSTM Sentiment Prediction:", prediction[0])
