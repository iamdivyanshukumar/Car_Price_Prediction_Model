# Car Price Prediction

This project is a machine learning-based web application that predicts the selling price of used cars. Given inputs such as make, model, year, mileage, fuel type, and transmission, it estimates a fair market value using a trained regression model. The application includes a complete data preprocessing pipeline and a Flask-based web interface.

---

## Overview

Used car markets often suffer from inconsistent listings and non-standard data. This project addresses those issues by building a robust model backed by thorough data cleaning and feature engineering. The result is a system that can predict car prices with a good degree of accuracy and explainability.

---

## Technical Summary

### Data Pipeline

1. **Data Cleaning**
   - Handles missing values for key fields like title status and accident history
   - Standardizes units in fields like mileage and price
   - Fixes common formatting and user input errors

2. **Feature Engineering**
   - Extracts numerical data from text fields, such as horsepower from engine descriptions
   - Identifies:
     - Number of cylinders
     - Transmission type
     - Fuel type
     - Standardized color groups
     - Vehicle age based on model year

3. **Preprocessing**
   - Numerical features are scaled and imputed
   - Categorical features are one-hot encoded
   - Ordinal features use special encoding based on domain knowledge

---

### Model

- **Algorithm**: Random Forest Regressor
- **Performance**:
  - R² Score: 0.84
  - RMSE: Approximately $2,450 (based on log-transformed prices)
- **Why Random Forest?**
  - Works well with non-linear relationships
  - Handles outliers gracefully
  - Provides built-in feature importance for explainability

---

## How to Use This Project

### 1. Running the Web Application

To use the web-based price predictor:

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Launch the Flask app:

    ```bash
    python app.py
    ```

3. Open your browser and go to `http://localhost:5000` to use the interface.

---

### 2. Training the Model

To retrain the model with your own dataset:

1. Prepare your data as `used_cars.csv`
2. Run the training script:

    ```bash
    python preprocessing.py
    ```

3. The trained model and preprocessing pipeline will be saved in the `model/` directory.

---

## Project Structure

```
car-price-prediction/
├── app.py                # Main Flask application
├── preprocessing.py      # Data cleaning and transformation logic
├── model/
│   ├── car_price_model.pkl     # Trained model
│   └── feature_columns.pkl  # Preprocessing pipeline
├── notebooks/
│   └── used_cars.csv   # Dataset
│   └── notebook.ipynb  # EDA and experimentation
├── static/
│   └── style.css         # Styling for web interface
├── templates/
│   └── index.html        # Web page template
└── requirements.txt      # Python dependencies
```

---

## Customization Guide

To adapt this project for your own car dataset:

- Update feature extraction and transformation logic in `preprocessing.py`
- Adjust the web form in `templates/index.html` to reflect your required inputs
- Modify the training script if your dataset structure differs

---

## Key Features

- End-to-end machine learning pipeline
- Fully integrated Flask web application
- Real-world data preprocessing
- Feature importance for explainability
- Easily customizable and extendable

---

## Model Evaluation

| Metric     | Value       |
|------------|-------------|
| R² Score   | 0.84        |
| RMSE       | $2,450      |
| Algorithm  | Random Forest Regressor |

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push to your branch
5. Submit a pull request for review

