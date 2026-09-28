import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained logistic regression model
model = joblib.load('Logistic Delay Assignment.sav')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Convert input data to DataFrame. Ensure it matches the training features.
        # This is a placeholder and assumes the input JSON keys match the original training features.
        # In a real application, you'd need more robust input validation and feature engineering.
        input_df = pd.DataFrame([data])

        # Assuming the input data 'data' contains only the numeric features used for 'logi'
        # The `x_train_processed` had 10 columns:
        # 'Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity',
        # 'Waiting_Time', 'User_Transaction_Amount', 'User_Purchase_Frequency',
        # 'Asset_Utilization', 'Demand_Forecast'

        # Ensure the columns in input_df match the features the model was trained on.
        # If feature names don't match, this will likely cause errors.
        # For example:
        expected_features = ['Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity',
                             'Waiting_Time', 'User_Transaction_Amount', 'User_Purchase_Frequency',
                             'Asset_Utilization', 'Demand_Forecast']
        input_df = input_df[expected_features] # Select and reorder columns if necessary

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_delay': float(prediction_proba[0][0]),
            'probability_delay': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
