from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
import os
from sklearn.preprocessing import OrdinalEncoder


app = Flask(__name__)

# Load the model and feature columns
model_path = os.path.join(os.path.dirname(__file__), 'model/car_price_model.pkl')
feature_columns_path = os.path.join(os.path.dirname(__file__), 'model/feature_columns.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please train the model first.")

try:
    with open(feature_columns_path, 'rb') as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    raise Exception("Feature columns file not found. Please train the model first.")

# Define all transformation functions
def extract_fuel_type(engine_info):
    if pd.isna(engine_info):
        return np.nan
    engine_info = str(engine_info)
    if 'Gasoline' in engine_info:
        return 'Gasoline'
    elif 'Hybrid' in engine_info:
        return 'Hybrid'
    elif 'Flex Fuel' in engine_info or 'E85' in engine_info:
        return 'Flex Fuel'
    elif 'Diesel' in engine_info:
        return 'Diesel'
    elif 'Electric' in engine_info:
        return 'Electric'
    else:
        return 'Other'

def extract_transmission_type(transmission):
    transmission = str(transmission)
    if 'Automatic' in transmission:
        return 'Automatic'
    elif 'Manual' in transmission:
        return 'Manual'
    elif 'CVT' in transmission:
        return 'CVT'
    elif 'DCT' in transmission:
        return 'DCT'
    elif 'Fixed Gear' in transmission:
        return 'Fixed Gear'
    elif 'Variable' in transmission:
        return 'Variable'
    elif 'Single-Speed' in transmission:
        return 'Single-Speed'
    else:
        return 'Other'

def categorize_color(color):
    if pd.isna(color):
        return 'Other'
    color = str(color).lower()
    color_categories = {
        'Black': ['black', 'obsidian', 'raven', 'onyx'],
        'White': ['white', 'pearl', 'ivory', 'frost'],
        'Blue': ['blue', 'navy', 'aqua', 'teal'],
        'Red': ['red', 'ruby', 'garnet'],
        'Silver_Gray': ['silver', 'gray', 'grey', 'steel'],
        'Green': ['green'],
        'Yellow_Orange': ['yellow', 'gold', 'orange']
    }
    for category, keywords in color_categories.items():
        if any(x in color for x in keywords):
            return category
    return 'Other'

def extract_hp(engine_string):
    if pd.isna(engine_string):
        return np.nan
    engine_string = str(engine_string)
    match = re.search(r'(\d+\.?\d*)HP', engine_string)
    if match:
        return float(match.group(1))
    return None

def extract_cylinders(engine_string):
    if pd.isna(engine_string):
        return np.nan
    engine_string = str(engine_string)
    match = re.search(r'(\d+)\s*Cylinder', engine_string)
    if match:
        return int(match.group(1))
    return None

def preprocess_input(input_data):
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Clean mileage and price
    df['milage'] = (
        df['milage']
        .str.replace('mi.', '', regex=False)
        .str.replace(',', '')
        .astype(float)
    )
    
    # Feature engineering
    df['engine_fuel_type'] = df['engine'].apply(extract_fuel_type)
    df['transmission_type'] = df['transmission'].apply(extract_transmission_type)
    df['ext_color'] = df['ext_col'].apply(categorize_color)
    df['int_color'] = df['int_col'].apply(categorize_color)
    
    # Extract numerical features
    df['horsepower'] = df['engine'].apply(extract_hp)
    df['cylinders'] = df['engine'].apply(extract_cylinders)
    
    # Calculate age
    current_year = datetime.now().year
    df['age'] = current_year - df['model_year']
    
    # Handle categorical encodings
    categories = {
        'fuel_type': ['E85 Flex Fuel', 'Gasoline', 'Hybrid', 'Diesel', 'Plug-In Hybrid'],
        'accident': ['None reported', 'At least 1 accident or damage reported'],
        'clean_title': ['No', 'Yes']
    }
    
    for column, cat_list in categories.items():
        encoder = OrdinalEncoder(categories=[cat_list])
        df[column] = encoder.fit_transform(df[[column]]).astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['transmission', 'model', 'ext_col', 'int_col', 'model_year', 'engine']
    df = df.drop(columns=columns_to_drop)
    
    # Ensure all expected columns are present
    for col in feature_columns['all_columns']:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training data
    df = df[feature_columns['all_columns']]
    
    return df

@app.route('/')
def home():
    current_year = datetime.now().year
    return render_template('index.html', current_year=current_year)

@app.route('/predict', methods=['POST'])
def predict():
    current_year = datetime.now().year
    try:
        # Get form data
        form_data = {
            'brand': request.form['brand'],
            'model': request.form['model'],
            'model_year': int(request.form['model_year']),
            'milage': request.form['milage'],
            'clean_title': request.form.get('clean_title', 'No'),
            'accident': request.form.get('accident', 'None reported'),
            'fuel_type': request.form['fuel_type'],
            'engine': request.form['engine'],
            'transmission': request.form['transmission'],
            'ext_col': request.form['ext_col'],
            'int_col': request.form['int_col']
        }
        
        # Preprocess the input
        processed_data = preprocess_input(form_data)
        
        # Make prediction
        log_price = model.predict(processed_data)[0]
        price = np.expm1(log_price)  # Convert back from log scale
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Price: ${price:,.2f}',
                             form_data=form_data,
                             current_year=current_year)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             form_data=request.form,
                             current_year=current_year)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        processed_data = preprocess_input(data)
        log_price = model.predict(processed_data)[0]
        price = np.expm1(log_price)
        return jsonify({'predicted_price': price})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()