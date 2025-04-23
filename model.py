import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re

# Load dataset
df = pd.read_csv('disease_symptoms_solutions_dataset.csv', encoding='Windows-1252')

# Train the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['symptoms'])
y = df['disease']
model = MultinomialNB()
model.fit(X, y)

def preprocess_symptoms(symptoms_text):
    """Preprocess symptoms input to better match dataset format"""
    # Convert to lowercase
    symptoms_text = symptoms_text.lower()
    
    # Remove any punctuation that isn't a comma (to keep symptom separation)
    symptoms_text = re.sub(r'[^\w\s,]', '', symptoms_text)
    
    # Ensure commas have spaces after them for consistent formatting
    symptoms_text = re.sub(r',(?! )', ', ', symptoms_text)
    
    return symptoms_text

def predict_disease(symptoms):
    """Predict disease based on symptoms input"""
    # Preprocess the symptoms text
    processed_symptoms = preprocess_symptoms(symptoms)
    
    # Vectorize the input
    symptoms_vectorized = vectorizer.transform([processed_symptoms])
    
    # Predict disease
    prediction = model.predict(symptoms_vectorized)[0]
    
    # Get top 3 predictions with probabilities
    probabilities = model.predict_proba(symptoms_vectorized)[0]
    top_indices = probabilities.argsort()[-3:][::-1]  # Get indices of top 3 predictions
    top_diseases = [(model.classes_[i], round(probabilities[i]*100, 2)) for i in top_indices]
    
    # Get solution for top prediction
    solution = df[df['disease'] == prediction]['solution'].values[0]
    
    return prediction, solution, top_diseases