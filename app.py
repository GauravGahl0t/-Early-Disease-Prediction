import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from PIL import Image
import http.client
import requests
import base64
import io
# import cv2
# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

selected = option_menu('Early Disease Prediction', [
    'Disease Prediction',
    'Lung Cancer Prediction',
],
    icons=['','activity', 'heart'],
    default_index=0,
    orientation="horizontal",
    )

# sidebar
with st.sidebar:


# Adding text to the sidebar
    st.sidebar.title("Welcome to the Disease Prediction Project")

    st.sidebar.write("-The project predicts what disease the patient might be suffering and how likely they are to have that disease. \n \n \n   -The user does not need to traverse different places in order to predict whether he/she has a particular disease or not.")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")
    st.sidebar.title("")




    st.sidebar.markdown("###   Akshaj Gupta")
    st.sidebar.markdown("###   Gaurav Gahlot")
    st.sidebar.markdown("###   Vikash Kumar")




# multiple disease prediction
if selected == 'Disease Prediction': 
    


    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')


# Load the dataset
lung_cancer_data = pd.read_csv('data/lung_cancer.csv')

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Lung Cancer Prediction')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_data=0
        gender="Upload image for Gender Detection"
        url = "https://age-detection-and-gender-detection-from-face-image.p.rapidapi.com/api/faces"
        headers = {
            "x-rapidapi-key": "bbc5145240msh0e91c37e7185464p106ea8jsn408663b4fe84",  # Replace with your actual RapidAPI key
            "x-rapidapi-host": "age-detection-and-gender-detection-from-face-image.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        # Streamlit app setup
       

        # Upload an image from local storage
        uploaded_file1 = st.file_uploader("Gender Detection", type=["jpg", "jpeg", "png"])

        if uploaded_file1 is not None:
            # Display the uploaded image
           

            # Convert the uploaded image to base64
            img_bytes = uploaded_file1.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Prepare the payload for the API request
            payload = {
                "base64_image": img_base64
            }

            # Send the request to the API
            response = requests.post(url, json=payload, headers=headers)

            # Display the response
            if response.status_code == 200:
                result = response.json()
                
                gender_data = result["models"][0]["output"][0]["gender"]
            else:
                st.write("Error: Upload another image")

        if gender_data!=0:
            if gender_data["Man"]>gender_data["Woman"]:
                gender="Male"
            else:
                gender="Female"
                
        st.write(gender)
        #gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
     


    with col2:
        url = "https://age-detection2.p.rapidapi.com/age"
        headers = {
            "x-rapidapi-key": "bbc5145240msh0e91c37e7185464p106ea8jsn408663b4fe84",  # Replace with your own key
            "x-rapidapi-host": "age-detection2.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        # Streamlit app setup
       

        # Image file uploader in Streamlit
        age="Upload image for age detection"
        uploaded_file2 = st.file_uploader("Age Detection", type=["jpg", "jpeg", "png"])

        if uploaded_file2 is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file2)

            # Convert the image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Prepare the payload for the API request
            payload = {
                "image": f"data:image/jpeg;base64,{img_str}",  # base64-encoded image
                "return_face": True
            }

            # Send request to RapidAPI endpoint
            response = requests.post(url, json=payload, headers=headers)

            # Display the response
            if response.status_code == 200:
                result = response.json()
                age=result.get("age")
                #st.write("Detected Age:", result.get("age", "N/A"))
            else:
                st.write("Error: Upload another image")
            


        st.write(str(age))
      
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:


            import streamlit as st
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.resnet50 import preprocess_input
            import numpy as np
            from PIL import Image

            # Load pre-trained model (e.g., ResNet50)
            model = ResNet50(weights='imagenet')  # Replace with your custom model if needed

            # Function to preprocess the image
            def preprocess_image(img):
                img = img.resize((224, 224))  # Resize image to match model input size
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = preprocess_input(img_array)  # Preprocess for ResNet50
                return img_array

            # Upload image
            uploaded_file = st.file_uploader("Anxiety Detection", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                # Open and display the uploaded image
                image = Image.open(uploaded_file)

                # Preprocess the image for model prediction
                img_array = preprocess_image(image)

                # Predict using the pre-trained model
                predictions = model.predict(img_array)
                decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0]

                # Example logic for "YES"/"NO" based on predictions
                anxiety_detected = False

                # Check if anxiety-related labels are predicted (e.g., 'fear', 'stress', etc.)
                for _, label, score in decoded_predictions:
                    if 'fear' in label.lower() or 'stress' in label.lower():  # You can customize this logic
                        anxiety_detected = True
                        break
                
                # Automatically update the select box based on the model prediction
                if anxiety_detected:
                    anxiety = 'YES'
                    st.write("Anxiety detected based on the image.")
                else:
                    anxiety = 'NO'
                    st.write("No anxiety detected based on the image.")

    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    cancer_result = ''

    # Button
    if st.button("Predict Lung Cancer"):
        # Create a DataFrame with user inputs
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # Map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()

        # Convert columns to numeric where necessary
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Perform prediction
        cancer_prediction = lung_cancer_model.predict(user_data)

        # Display result
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + cancer_result)




