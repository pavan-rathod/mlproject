from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Fetch and clean form data
        gender = request.form.get('gender', '').strip()
        race_ethnicity = request.form.get('race_ethnicity', '').strip()
        parental_level_of_education = request.form.get('parental_level_of_education', '').strip()
        lunch = request.form.get('lunch', '').strip()
        test_preparation_course = request.form.get('test_preparation_course', '').strip()
        writing_score = request.form.get('writing_score', '').strip()
        reading_score = request.form.get('reading_score', '').strip()

        # Validate all fields
        if '' in [gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, writing_score, reading_score]:
            return render_template('home.html', results="⚠️ Please fill in all fields correctly.")

        try:
            # Create input data object
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                writing_score=float(writing_score),
                reading_score=float(reading_score)
            )

            # Convert to DataFrame and predict
            pred_df = data.get_data_as_data_frame()
            print("User Input:\n", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=f"Predicted Math Score: {results[0]:.2f}")

        except Exception as e:
            return render_template('home.html', results=f"❌ Error: {str(e)}")

    else:
        return render_template('home.html')
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
