from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Correct path: Look inside the 'model' folder
model_path = os.path.join('model', 'titanic_survival_model.pkl')

# Load Model
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Error: Model not found at {model_path}. Run model/model_building.py first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model is missing.")

    try:
        # 1. Retrieve inputs
        pclass = int(request.form['Pclass'])
        sex = int(request.form['Sex']) # 0 for Male, 1 for Female
        age = float(request.form['Age'])
        fare = float(request.form['Fare'])

        # 2. Prepare features
        features = np.array([[pclass, sex, age, fare]])

        # 3. Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        # 4. Format Output
        if prediction == 1:
            result_text = f"SURVIVED (Probability: {probability:.1f}%)"
            css_class = "safe"
        else:
            result_text = f"DID NOT SURVIVE (Probability: {probability:.1f}%)"
            css_class = "danger"

        return render_template('index.html', 
                             prediction_text=result_text, 
                             result_class=css_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)