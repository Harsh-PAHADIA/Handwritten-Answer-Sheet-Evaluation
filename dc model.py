import sys
import os
import pymysql
import cv2
import re
import requests
import string
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('stopwords')

# Connect to MySQL database
db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="aep"
)
cursor = db.cursor()

# Get student USN from command line argument
student_usn = sys.argv[1]

# Fetch student's answer records and key answer + keywords from database
cursor.execute("SELECT * FROM answer WHERE usn=%s", student_usn)
student_data = cursor.fetchall()
cursor.execute("SELECT key_answer, keywords FROM key_answers WHERE usn=%s", student_usn)
key_answer_data = cursor.fetchone()
key_answer = key_answer_data[0]
keywords = key_answer_data[1].split(',')

# OCR to extract text from image
def get_text_from_image(file_to_ocr):
    im = Image.open(file_to_ocr)
    txt = pytesseract.image_to_string(im)
    return txt

# Text preprocessing
def preprocess_sentence(stmt):
    stmt = stmt.lower().translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word for word in nltk.word_tokenize(stmt) if word not in stop_words]
    return " ".join(words)

# Cosine similarity calculation
def calculate_cosine_similarity(student_answer, key_answer):
    vectorizer = TfidfVectorizer().fit_transform([student_answer, key_answer])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1] * 100  # Cosine similarity in percentage

# Grammar and spelling check using TextGears API
def grammar_check(sentence):
    response = requests.get("https://api.languagetool.org/v2/check", params={
        'text': sentence,
        'language': 'en-IN'
    })
    errors = response.json().get('matches', [])
    count_spelling = sum(1 for error in errors if error['rule']['category']['id'] == 'TYPOS')
    count_grammar = len(errors) - count_spelling  # Remaining errors are considered grammar issues
    return count_spelling, count_grammar


# Keyword matching
def keyword_match(student_answer, keywords):
    return [word for word in keywords if word in student_answer]

# Extract features for student answers
for n, data in enumerate(student_data):
    path = os.path.join("MachineLearning/answerimage", str(data[0]))
    os.makedirs(path, exist_ok=True)
    
    for j, answer_image in enumerate(data[1:]):
        if answer_image:
            image_text = get_text_from_image(os.path.join("C:/xampp1/htdocs/WebApp/AEP", answer_image))
            preprocessed_text = preprocess_sentence(image_text)
            word_count, sentence_count = len(nltk.word_tokenize(preprocessed_text)), len(nltk.sent_tokenize(preprocessed_text))
            cosine_sim = calculate_cosine_similarity(preprocessed_text, preprocess_sentence(key_answer))
            spelling_errors, grammar_errors = grammar_check(image_text)
            matched_keywords = keyword_match(preprocessed_text, keywords)
            
            # Insert into database
            cursor.execute("""
                INSERT INTO extracted_parameters 
                (usn, word_count, sentence_count, cosine_similarity, spelling_errors, grammar_errors, matched_keywords_count) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (student_usn, word_count, sentence_count, cosine_sim, spelling_errors, grammar_errors, len(matched_keywords)))

db.commit()

# Fetch extracted parameters for model training
cursor.execute("SELECT word_count, sentence_count, cosine_similarity, spelling_errors, grammar_errors, matched_keywords_count, grade FROM extracted_parameters")
data = cursor.fetchall()
df = pd.DataFrame(data, columns=['word_count', 'sentence_count', 'cosine_similarity', 'spelling_errors', 'grammar_errors', 'matched_keywords_count', 'grade'])

# Split data into features and target
X = df[['word_count', 'sentence_count', 'cosine_similarity', 'spelling_errors', 'grammar_errors', 'matched_keywords_count']]
y = df['grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Assuming grades are on a continuous scale
])

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Mean Absolute Error: {mae}")

# Make predictions
predictions = model.predict(X_test)
for i, pred in enumerate(predictions[:10]):  # Display the first 10 predictions
    print(f"Predicted Grade: {pred[0]}, Actual Grade: {y_test.iloc[i]}")

# Close database connection
db.close()
