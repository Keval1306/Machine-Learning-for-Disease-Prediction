import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit page configuration
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Load and prepare the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Training.csv")
    return data

data = load_data()

# Prepare features and target variable
X = data.iloc[:, :-1]
y = data['prognosis']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# One-hot encode features
X_encoded = pd.get_dummies(X)

# Add columns for age and gender, initialized to 0
X_encoded['age'] = 0  
X_encoded['gender'] = 0  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit UI components
st.title("Disease Prediction System")
st.write("### Provide your symptoms and personal information to predict the disease.")
st.markdown("___")

# User input for age and gender
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=25)

with col2:
    gender = st.selectbox("Select your gender:", ("Male", "Female", "Other"))

st.markdown("### Select Your Symptoms")
st.write("Please select all applicable symptoms:")

# Symptom selection
symptom_columns = X_encoded.columns[:-2]  # Exclude age and gender

selected_symptoms_selectbox = st.multiselect(
    "Search and select your symptoms:",
    options=symptom_columns.tolist(),
    default=[],
    help="Start typing to search for your symptoms."
)

st.write("Or select symptoms using checkboxes:")
selected_symptoms_checkboxes = []
for symptom in symptom_columns:
    if st.checkbox(symptom, value=False):
        selected_symptoms_checkboxes.append(symptom)

# Combine selected symptoms from both inputs
selected_symptoms = list(set(selected_symptoms_selectbox) | set(selected_symptoms_checkboxes))

# Handle case where no symptoms are selected
if not selected_symptoms:
    st.warning("Please select at least one symptom.")
else:
    # Prepare input data for prediction
    input_data = np.zeros((1, len(symptom_columns)))  # Initialize input_data with zeros

    # Fill the input_data based on selected symptoms
    for symptom in selected_symptoms:
        if symptom in symptom_columns:  # Check if the symptom is valid
            input_data[0, np.where(symptom_columns == symptom)[0][0]] = 1  # Set the corresponding index to 1

    # Encode gender
    gender_encoded = np.array([[1 if gender == "Male" else (0 if gender == "Female" else -1)]])
    input_data = np.concatenate((input_data, gender_encoded), axis=1)

    # Add age to the input data
    age_input = np.array([[age]])
    input_data = np.concatenate((input_data, age_input), axis=1)

    # Prediction on button click
    if st.button('Predict', key='predict_btn'):
        try:
            prediction = clf.predict(input_data)
            predicted_disease = le.inverse_transform(prediction)  # Decode back to the disease name
            st.success(f"**Predicted Disease:** {predicted_disease[0]}")
        except ValueError as e:
            st.error(f"Error in prediction: {e}")

# Show model accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write("### Model Performance")
st.sidebar.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
