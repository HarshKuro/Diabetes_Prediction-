# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (using Pima Indians Diabetes Dataset as an example)
def load_data():
    # For simplicity, we'll use the Pima Indians Diabetes Dataset
    # Load data from a CSV file or similar
    data = pd.read_csv(r"C:\Users\Kuro7\Documents\Medical diabities prediction\diabetes.csv")
    return data

# Prepare data for training
def prepare_data(data):
    X = data.drop(columns=['Outcome'])  # Features (exclude target column)
    y = data['Outcome']  # Target column (1 = Diabetic, 0 = Non-Diabetic)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Train model
def train_model(X_train, y_train):
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model to a file for future use
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load the trained model from file
def load_model():
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Predict diabetes based on user input
def predict_diabetes(model, scaler, input_data):
    # Convert input data to a NumPy array and reshape for the model
    input_array = np.array(input_data).reshape(1, -1)
    
    # Standardize the input data
    standardized_input = scaler.transform(input_array)
    
    # Make prediction (1 = Diabetic, 0 = Non-Diabetic)
    prediction = model.predict(standardized_input)
    return prediction[0]

# Get user input for prediction
def get_user_input():
    print("Please enter the following health parameters:")
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    # Return the inputs as a list
    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# Main function to run the prediction
def main():
    # Load the data and prepare it
    data = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    
    # Train the model if not already trained
    train_model(X_train, y_train)
    
    # Load the trained model
    model = load_model()
    
    # Get input from the user
    user_input = get_user_input()
    
    # Predict whether the user is diabetic
    prediction = predict_diabetes(model, scaler, user_input)
    
    # Output the prediction result
    if prediction == 1:
        print("The patient is likely diabetic.")
    else:
        print("The patient is not diabetic.")

# Run the prediction script
if __name__ == '__main__':
    main()
