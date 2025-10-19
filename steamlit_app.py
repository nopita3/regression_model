import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
filename = 'model-reg-67130700320.pkl' # Make sure this filename matches the saved model
loaded_model = pickle.load(open(filename, 'rb'))

st.title('Sales Prediction App')

tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

with tab1:
    st.header("Manual Input")

    # Function to get user input
    def user_input_features():
        col1, col2, col3 = st.columns(3)
        with col1:
            youtube = st.text_input('YouTube', '100.0')
        with col2:
            tiktok = st.text_input('TikTok', '25.0')
        with col3:
            instagram = st.text_input('Instagram', '50.0')

        data = {'youtube': float(youtube),
                'tiktok': float(tiktok),
                'instagram': float(instagram)}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # st.subheader('User Input features') # Commented out as requested
    # st.write(input_df) # Commented out as requested

    # Predict using the loaded model
    prediction = loaded_model.predict(input_df.to_numpy())

    st.subheader('Prediction (Sales)')
    st.write(prediction)

with tab2:
    st.header("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        # st.subheader('Uploaded Data') # Commented out as requested
        # st.write(batch_df) # Commented out as requested

        # Predict on the uploaded data
        batch_prediction = loaded_model.predict(batch_df.to_numpy())

        st.subheader('Batch Predictions (Sales)')
        batch_df['predicted_sales'] = batch_prediction
        st.write(batch_df)
