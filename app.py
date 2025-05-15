# prompt: using these models file create a gui interface and streamlit app

!pip install streamlit

%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the models
svm_model = joblib.load('svm_model.pkl')
nb_model = joblib.load('nb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
lr_model = joblib.load('lr_model.pkl')
knn_model = joblib.load('knn_model.pkl')

# Create the Streamlit app
st.title('Social Network Ad Purchase Prediction')

st.sidebar.header('User Input Features')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 18, 65, 30)
    estimated_salary = st.sidebar.slider('Estimated Salary', 15000, 150000, 60000)
    
    gender_encoded = 1 if gender == 'Male' else 0

    data = {'Gender': gender_encoded,
            'Age': age,
            'EstimatedSalary': estimated_salary}
    features = np.array([list(data.values())])
    return features

input_data = user_input_features()

st.subheader('User Input')
st.write(input_data)

# Make predictions
models = {
    "SVM": svm_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "KNN": knn_model
}

st.subheader('Prediction Results')

for model_name, model in models.items():
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None

    st.write(f'**{model_name} Prediction:**')
    if prediction[0] == 1:
        st.success('Will Purchase the Ad')
    else:
        st.error('Will Not Purchase the Ad')

    if prediction_proba is not None:
        st.write(f'Probability of Not Purchasing: {prediction_proba[0][0]:.4f}')
        st.write(f'Probability of Purchasing: {prediction_proba[0][1]:.4f}')