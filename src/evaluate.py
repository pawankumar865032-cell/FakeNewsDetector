import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# Load model and vectorizer
model = joblib.load("../model/fake_news_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")

# Load datasets
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine data
df = pd.concat([fake, true], ignore_index=True)

X = df["text"]
y = df["label"]

# Transform text
X = vectorizer.transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()