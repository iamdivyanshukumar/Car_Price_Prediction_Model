import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    StandardScaler,
    FunctionTransformer
)
import os
from sklearn.ensemble import RandomForestRegressor
import pickle
import re

# 1. Define all your custom transformation functions
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
    if any(x in color for x in ['black', 'obsidian', 'raven', 'onyx']):
        return 'Black'
    elif any(x in color for x in ['white', 'pearl', 'ivory', 'frost']):
        return 'White'
    elif any(x in color for x in ['blue', 'navy', 'aqua', 'teal']):
        return 'Blue'
    elif any(x in color for x in ['red', 'ruby', 'garnet']):
        return 'Red'
    elif any(x in color for x in ['silver', 'gray', 'grey', 'steel']):
        return 'Silver_Gray'
    elif any(x in color for x in ['green']):
        return 'Green'
    elif any(x in color for x in ['yellow', 'gold', 'orange']):
        return 'Yellow_Orange'
    else:
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

def preprocess_data(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Clean mileage and price - use direct assignment instead of inplace
    df['milage'] = df['milage'].str.replace('mi.', '', regex=False).str.replace(',', '')
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    
    df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Fill missing values - use direct assignment
    df['clean_title'] = df['clean_title'].fillna('No')
    df['accident'] = df['accident'].fillna('None reported')
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], np.nan)
    
    # Feature engineering
    df['engine_fuel_type'] = df['engine'].apply(extract_fuel_type)
    df['transmission_type'] = df['transmission'].apply(extract_transmission_type)
    df['ext_color'] = df['ext_col'].apply(categorize_color)
    df['int_color'] = df['int_col'].apply(categorize_color)
    
    # Extract numerical features
    df['horsepower'] = df['engine'].apply(extract_hp)
    df['cylinders'] = df['engine'].apply(extract_cylinders)
    
    # Calculate age
    max_year = df['model_year'].max()
    df['age'] = max_year - df['model_year']
    
    # Drop unnecessary columns - use direct assignment
    columns_to_drop = ['transmission', 'model', 'ext_col', 'int_col', 'model_year', 'engine']
    df = df.drop(columns=columns_to_drop)
    
    return df

def main():
    # 2. Load your data
    try:
        df = pd.read_csv('notebooks/used_cars.csv')
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found at 'notebooks/used_cars.csv'")

    # 3. Apply initial preprocessing
    df = preprocess_data(df)

    # 4. Define features and target
    X = df.drop('price', axis=1)
    y = np.log1p(df['price'])  # Log transform the target

    # 5. Define column types
    numeric_features = ['milage', 'horsepower', 'cylinders', 'age']
    categorical_features = ['brand', 'engine_fuel_type', 'transmission_type', 
                          'ext_color', 'int_color']
    ordinal_features = ['fuel_type', 'accident', 'clean_title']

    # 6. Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 7. Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    # 8. Create full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 9. Train the model
    model.fit(X, y)

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # 10. Save the entire pipeline (including preprocessing)
    with open('model/car_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 11. Save the column names for later reference during inference
    feature_columns = {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': ordinal_features,
        'all_columns': X.columns.tolist()
    }

    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

    print("Model training and saving completed successfully!")

if __name__ == '__main__':
    main()