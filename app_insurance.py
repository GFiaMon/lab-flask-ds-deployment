# ==================== FLASK APP FOR INSURANCE MODEL DEPLOYMENT =======================

from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

import os

# Use a meaningful template directory name
app = Flask(__name__, template_folder='frontend')

# Load the trained model and model info from models directory
models_dir = 'models'
model_path = os.path.join(models_dir, 'insurance_model.pkl')
model_info_path = os.path.join(models_dir, 'model_info.pkl')

model = pickle.load(open(model_path, 'rb'))
model_info = pickle.load(open(model_info_path, 'rb'))

print("Model loaded successfully!")
print(f"Features: {model_info['feature_names']}")
print(f"Test R²: {model_info['test_r2']}")

# Feature engineering function (same as in training)
def prepare_features(input_data):
    """
    Prepare features for prediction - same as used in training
    """
    df_processed = pd.DataFrame([input_data])
    
    # Create BMI categories
    df_processed['bmi_category'] = pd.cut(df_processed['bmi'], 
                                        bins=[0, 18.5, 25, 30, 100], 
                                        labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[0, 30, 45, 60, 100], 
                                     labels=['young', 'adult', 'middle_aged', 'senior'])
    
    # Interaction features
    df_processed['smoker_age_interaction'] = df_processed['smoker'].map({'yes': 1, 'no': 0}) * df_processed['age']
    df_processed['smoker_bmi_interaction'] = df_processed['smoker'].map({'yes': 1, 'no': 0}) * df_processed['bmi']
    
    return df_processed

@app.route('/')
def home():
    return render_template('index.html', 
                         feature_names=model_info['feature_names'],
                         numerical_features=model_info['numerical_features'],
                         categorical_features=model_info['categorical_features'],
                         model_r2=model_info['test_r2'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data for the original features
        input_data = {
            'age': float(request.form['age']),
            'sex': request.form['sex'],
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': request.form['smoker'],
            'region': request.form['region']
        }
        
        print("Raw input:", input_data)
        
        # Prepare features using the same transformation as training
        processed_features = prepare_features(input_data)
        print("Processed features shape:", processed_features.shape)
        print("Processed features columns:", processed_features.columns.tolist())
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        print("Raw prediction:", prediction)
        
        # Prepare results
        results = {
            'prediction': f"${prediction:,.2f}",
            'input_features': input_data,
            'model_r2': f"{model_info['test_r2']:.4f}"
        }
        
        return render_template('index.html',
                             feature_names=model_info['feature_names'],
                             numerical_features=model_info['numerical_features'],
                             categorical_features=model_info['categorical_features'],
                             model_r2=model_info['test_r2'],
                             prediction_result=results)
    
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        print("Error:", error_msg)
        return render_template('index.html',
                             feature_names=model_info['feature_names'],
                             numerical_features=model_info['numerical_features'],
                             categorical_features=model_info['categorical_features'],
                             model_r2=model_info['test_r2'],
                             error=error_msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        input_data = {
            'age': float(data['age']),
            'sex': data['sex'],
            'bmi': float(data['bmi']),
            'children': int(data['children']),
            'smoker': data['smoker'],
            'region': data['region']
        }
        
        # Prepare features using the same transformation as training
        processed_features = prepare_features(input_data)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        return jsonify({
            'predicted_charges': float(prediction),
            'predicted_charges_formatted': f"${prediction:,.2f}",
            'input_features': input_data,
            'model_r2': model_info['test_r2'],
            'model_parameters': model_info['best_params']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask application for Insurance Charges Prediction...")
    print("Model loaded successfully!")
    print(f"Features: {model_info['feature_names']}")
    print(f"Numerical features: {model_info['numerical_features']}")
    print(f"Categorical features: {model_info['categorical_features']}")
    print(f"Model R² score: {model_info['test_r2']:.4f}")
    app.run(debug=True, host='0.0.0.0', port=5001)




# ==================== FLASK APP Template from example =======================

# from flask import Flask, request, render_template, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and model info
# model = pickle.load(open('model.pkl', 'rb'))
# print("Model loaded successfully:", model)
# model_info = pickle.load(open('model_info.pkl', 'rb'))
# print("Model info loaded successfully:", model_info)

# @app.route('/')
# def home():
#     return render_template('index.html', 
#                          feature_names=model_info['feature_names'],
#                          target_names=model_info['target_names'])

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         features = []
#         feature_names = model_info['feature_names']
        
#         for feature_name in feature_names:
#             # Convert feature names to form field names (replace spaces and special chars)
#             form_field = feature_name.replace(' ', '_').replace('(', '').replace(')', '')
#             features.append(float(request.form[form_field]))
        
#         # Make prediction
#         prediction = model.predict(np.array(features).reshape(1, -1))
#         prediction_proba = model.predict_proba(np.array(features).reshape(1, -1))
        
#         # Get the predicted class name
#         predicted_class = model_info['target_names'][prediction[0]]
#         confidence = max(prediction_proba[0]) * 100
        
#         # Prepare results
#         results = {
#             'prediction': predicted_class,
#             'confidence': f"{confidence:.2f}%",
#             'input_features': dict(zip(feature_names, features))
#         }
        
#         return render_template('index.html',
#                              feature_names=feature_names,
#                              target_names=model_info['target_names'],
#                              prediction_result=results)
    
#     except Exception as e:
#         error_msg = f"Error making prediction: {str(e)}"
#         return render_template('index.html',
#                              feature_names=model_info['feature_names'],
#                              target_names=model_info['target_names'],
#                              error=error_msg)

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """API endpoint for predictions"""
#     try:
#         data = request.json
#         features = [data[f'feature_{i}'] for i in range(4)]
        
#         prediction = model.predict(np.array(features).reshape(1, -1))
#         prediction_proba = model.predict_proba(np.array(features).reshape(1, -1))
        
#         return jsonify({
#             'prediction': model_info['target_names'][prediction[0]],
#             'confidence': float(max(prediction_proba[0])),
#             'probabilities': dict(zip(model_info['target_names'], prediction_proba[0].tolist()))
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     print("Starting Flask application...")
#     print("Model loaded successfully!")
#     print(f"Features: {model_info['feature_names']}")
#     print(f"Classes: {model_info['target_names']}")
#     app.run(debug=True, host='0.0.0.0', port=5001)


