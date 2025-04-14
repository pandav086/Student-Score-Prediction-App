import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Ashu:ashu1234@cluster0.a7il5wn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db["student_pred"]


def load_model():
    with open("student_lr_final_model.pkl",'rb') as file:
        model,scaler,le= pickle.load(file)
    return model,scaler,le


def preprocessing_input_data(data,scaler,le):
   data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
   df = pd.DataFrame([data])
   df_transformed = scaler.transform(df)
   return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.title("student performance prediction")
    st.write("enter your data to get a prediction for your performance")

    hour_studied = st.number_input('Hour Studied',max_value= 10 ,min_value= 1, value= 5)
    previous_scores = st.number_input('Previous Scores',max_value= 100 ,min_value= 40, value= 70)
    Extra_activities = st.selectbox('Extracurricular Activities', ['Yes','No'])
    sleep_hours = st.number_input('Sleep Hours',max_value= 10 ,min_value= 4, value= 7)
    sample_paper_practiced = st.number_input('Sample Question Papers Practiced',max_value= 10 ,min_value= 0, value= 5)


    if st.button('predict your score'):
        user_data = {
            "Hours Studied" : hour_studied,
            "Previous Scores" : previous_scores,
            "Extracurricular Activities" : Extra_activities,
            "Sleep Hours" : sleep_hours,
            "Sample Question Papers Practiced" : sample_paper_practiced
        }

        prediction = predict_data(user_data)
        st.success(f"your prediction result is {prediction}")
        user_data['prediction'] = round(float(prediction[0]),2)
        user_data = {key : int(value) if isinstance(value , np.integer) else float(value) if isinstance(value , np.floating) else value for key,value in user_data.items()}
        collection.insert_one(user_data)
        

if __name__ == "__main__":
    main()    
