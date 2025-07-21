from flask import Flask, request, render_template
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app = application

# Route for a home page
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Fetch form data
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        writing_score = request.form.get('writing_score')
        reading_score = request.form.get('reading_score')

        # Validate input
        if None in [gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, writing_score, reading_score]:
            return render_template('home.html', results="Invalid input. Please fill all fields.")

        # Create data object
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            writing_score=float(writing_score),
            reading_score=float(reading_score)
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
