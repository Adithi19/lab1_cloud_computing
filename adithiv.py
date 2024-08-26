from flask import Flask, request, render_template
import numpy as np
from sklearn import linear_model

app = Flask(__name__)

# Create the linear regression model
reg = linear_model.LinearRegression()

# Training data
X = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0]]
y = [8, 10, 12, 14, 16, 18, 20, 22]

# Train the model
reg.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input
        user_input = request.form['user_input']
        
        # Convert the input to a float
        user_input = float(user_input)
        
        # Make a prediction using the model
        prediction = reg.predict([[user_input]])
        
        return f"Predicted value: {prediction[0]:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
