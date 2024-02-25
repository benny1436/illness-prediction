import pandas as pd
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv("illness-dataset - cleaned.csv")

# Clean data
def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

data['cleaned_symptoms'] = data['Symptoms'].apply(clean_text)

# Prepare data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_symptoms'])
encoder = LabelEncoder()
y = encoder.fit_transform(data['Disease'])
# comment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train.toarray(), y_train, epochs=100, batch_size=25, validation_split=0.1)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test.toarray()), axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Save the model
model.save("dish_prediction_model")

# # Example prediction
new_ingredients = ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"]

new_ingredients_vector = vectorizer.transform([clean_text(" ".join(new_ingredients))])
predicted_index = np.argmax(model.predict(new_ingredients_vector), axis=1)
predicted_dish = encoder.inverse_transform(predicted_index)

print("Predicted disease:", predicted_dish[0])