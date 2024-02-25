from flask import Flask, request, jsonify
app = Flask(__name__)
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
@app.route("/request", methods=["GET"])
def request_get():
    model2 = tf.keras.models.load_model('dish_prediction_model')

    extra = request.args.get("extra")
    new_ingredients = extra.split(", ")
    
    new_ingredients_vector = vectorizer.transform([clean_text(" ".join(new_ingredients))])

    predicted_index = np.argmax(model2.predict(new_ingredients_vector), axis=1)
    predicted_dish = encoder.inverse_transform(predicted_index)

    print("Predicted disease:", predicted_dish[0])

    return predicted_dish[0]


if __name__ == "__main__":
    app.run(debug=True)
