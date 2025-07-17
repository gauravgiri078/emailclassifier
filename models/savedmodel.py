import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Load and prepare data
df = pd.read_csv(r'C:\Users\gaura\OneDrive\Desktop\aiassignment\data\mail_data.csv')
data = df.where(pd.notnull(df), "")
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X = data['Message']
Y = data['Category'].astype('int')

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_features, Y_train)

# Evaluate model
y_pred = model.predict(X_test_features)
y_proba = model.predict_proba(X_test_features)[:, 1]

print("Classification Report:\n")
print(classification_report(Y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(Y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

# Save model and vectorizer
model_dir = r'C:\Users\gaura\OneDrive\Desktop\aiassignment\models'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'naive_bayes_model.joblib')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\nModel saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

# CLI for prediction
while True:
    message = input("\nEnter a message to classify (or type 'exit' to quit): ")
    if message.lower() == 'exit':
        break
    input_data = vectorizer.transform([message])
    prediction = model.predict(input_data)[0]
    print("Prediction:", "Ham" if prediction == 1 else "Spam")
