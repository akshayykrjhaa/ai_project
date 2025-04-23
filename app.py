from flask import Flask, render_template, request
from model import predict_disease
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, solution, top_diseases = None, None, None
    
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        prediction, solution, top_diseases = predict_disease(symptoms)
    
    # Load common symptoms list for suggestions
    df = pd.read_csv('disease_symptoms_solutions_dataset.csv', encoding='Windows-1252')
    all_symptoms = []
    for symptom_list in df['symptoms']:
        symptoms = [s.strip() for s in str(symptom_list).split(',')]
        all_symptoms.extend(symptoms)
    
    # Get unique symptoms and sort them
    unique_symptoms = sorted(list(set([s for s in all_symptoms if s and len(s) > 2])))
    
    return render_template('as.html', 
                          prediction=prediction, 
                          solution=solution, 
                          top_diseases=top_diseases,
                          symptoms_list=unique_symptoms)

if __name__ == '__main__':
    app.run(debug=True)