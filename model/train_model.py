import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# load dataset
df = pd.read_csv("dataset/depression_dataset_reddit_cleaned.csv")

print("Dataset loaded successfully\n")

# separate text and labels
texts = df["clean_text"]
labels = df["is_depression"]

# convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

print("Text converted to vectors")
print("Shape:", X.shape)

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training complete")

# test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)



# save model and vectorizer
joblib.dump(model, "model/depression_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully")

# test with a custom sentence
test_text = [
    "I feel exhausted and nothing excites me anymore"
]

test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("\nPrediction: Depression signal detected")
else:
    print("\nPrediction: No depression signal detected")