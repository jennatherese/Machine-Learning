from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model
model = joblib.load('final_model.pkl')

# Create app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        absences = float(request.form['absences'])
        goout = float(request.form['goout'])

        avg_grade = (G1 + G2) / 2
        engagement_score = studytime - goout

        # Make prediction
        features = np.array([[G1, G2, studytime, failures, absences, goout, avg_grade, engagement_score]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"🎓 Predicted Final Grade (G3): {prediction:.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text="❌ Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
