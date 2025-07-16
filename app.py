from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the saved model
model = joblib.load('C:/Users/RAHUL/OneDrive/Desktop/codes/ML/Titanic/logistic_regression_model.pkl')

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        pclass = int(request.form['Pclass'])
        sex = int(request.form['Sex'])  # 0 for male, 1 for female
        age = float(request.form['Age'])
        
        # Prepare the data for prediction
        input_data = np.array([[pclass, sex, age]])
        
        # Predict using the model
        prediction = model.predict(input_data)
        
        # Return result as string
        if prediction[0] == 1:
            result = "Survived"
        else:
            result = "Did not survive"
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
