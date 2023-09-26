from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import webbrowser

app = Flask(__name__)

model = joblib.load('data and models/cat_boost_regressor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

def apply_condition_adjustment(condition, repainted, predicted_value):
    adjustment_percentage = {
        (10, 'No'): 0, (9, 'No'): 0.03, (8, 'No'): 0.05, (7, 'No'): 0.07, (6, 'No'): 0.09,
        (5, 'No'): 0.11, (4, 'No'): 0.13, (3, 'No'): 0.15, (2, 'No'): 0.17, (1, 'No'): 0.19,
        (10, 'Yes'): 0.05, (9, 'Yes'): 0.07, (8, 'Yes'): 0.09, (7, 'Yes'): 0.11, (6, 'Yes'): 0.13,
        (5, 'Yes'): 0.15, (4, 'Yes'): 0.17, (3, 'Yes'): 0.19, (2, 'Yes'): 0.21, (1, 'Yes'): 0.23
    }

    key = (condition, repainted)
    if key in adjustment_percentage:
        adjustment = adjustment_percentage[key]
        adjusted_value = predicted_value - (predicted_value * adjustment)
        return adjusted_value
    else:
        return predicted_value

@app.route('/predict', methods=['POST'])
def predict():
    try:
        car_name = request.form['car_name']
        model_year = int(request.form['model_year'])
        manufacturer = request.form['manufacturer']
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']
        engine_cc = int(request.form['engine_cc'])
        kms_driven = int(request.form['kms_driven'])
        condition = int(request.form['condition'])
        repainted = request.form['repainted']

        data = {
            'car_name': [car_name],
            'model_year': [model_year],
            'manufacturer': [manufacturer],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'engine_cc': [engine_cc],
            'kms_driven': [kms_driven],
            'condition': [condition],
            'repainted': [repainted]
        }

        df = pd.DataFrame(data)
        prediction_data = df.drop(columns=['condition', 'repainted'])
        prediction = model.predict(prediction_data)[0]
        adjusted_prediction = apply_condition_adjustment(condition, repainted, prediction)
        return jsonify({'prediction': adjusted_prediction})

    except Exception as e:
        return jsonify({'error_message': 'Data not found'})

if __name__ == '__main__':
    webbrowser.open_new('http://localhost:5000')
    app.run(debug=True, use_reloader=False, port=5000, host='localhost')
