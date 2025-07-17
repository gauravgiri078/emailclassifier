
---

# 📧 Spam Email Classifier

A simple yet effective Spam Email Classifier built using **Python**, **scikit-learn**, and **Natural Language Processing (NLP)** techniques. This project uses a **Multinomial Naive Bayes** algorithm with **TF-IDF vectorization** to detect whether a message is spam or ham (not spam).

## 🧠 Features

* Preprocesses a dataset of labeled messages
* Converts raw text into TF-IDF features
* Trains a Naive Bayes classifier
* Evaluates the model with:

  * Classification report
  * Confusion matrix
  * ROC curve and AUC score
* Includes a CLI for live message prediction

## 📁 Dataset

The dataset (`mail_data.csv`) should be placed at:

```
C:\Users\gaura\OneDrive\Desktop\aiassignment\data\mail_data.csv
```

The CSV is expected to have two columns:

* **Category**: 'spam' or 'ham'
* **Message**: the email/message text

## 🛠️ Requirements

Install required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. **Train and Evaluate the Model**

   The script:

   * Loads and cleans the dataset
   * Converts text using TF-IDF vectorization
   * Trains a Multinomial Naive Bayes classifier
   * Evaluates model performance (accuracy, confusion matrix, ROC curve)

2. **Interactive CLI for Message Prediction**

   After training, you can enter messages manually to classify them as **Spam** or **Ham**.

## 📊 Output Examples

* **Confusion Matrix**

  Visualizes the true positives, false positives, etc.

* **ROC Curve**

  Shows the model’s ability to distinguish between classes, with AUC score.

* **Classification Report**

  Provides precision, recall, f1-score, and support.

## 🧪 Sample Usage

```bash
Enter a message to classify (or type 'exit' to quit): 
Win a free iPhone now!!!

Prediction: Spam
```

## 🧼 Notes

* `Category` labels are converted to:

  * `0` for Spam
  * `1` for Ham

* Model uses:

  * `TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)`
  * `MultinomialNB()` for classification

## 📌 To-Do

* Add model persistence with `joblib` or `pickle`
* Create a web interface using Flask or Streamlit
* Test with larger or more diverse datasets

---

