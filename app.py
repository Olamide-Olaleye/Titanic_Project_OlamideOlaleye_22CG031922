from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained Titanic model (Renamed)
model_filename = 'titanic_survival_model.pkl'

if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
else:
    # Error message updated to reflect new script name 'model_building.py'
    print(f"Error: '{model_filename}' not found. Please run 'python model_building.py' first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model not loaded.')

    try:
        # Extract data from form
        pclass = int(request.form['Pclass'])
        sex = int(request.form['Sex']) # 0 for male, 1 for female
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])

        # Create array for the model
        features = np.array([[pclass, sex, age, sibsp, parch, fare]])

        # Make prediction
        prediction = model.predict(features)

        # 0 = Did not survive, 1 = Survived
        output = 'Survived' if prediction[0] == 1 else 'Did Not Survive'

        return render_template('index.html', prediction_text='Prediction: The passenger likely {}'.format(output))

    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)