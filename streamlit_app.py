import streamlit as st
import pandas as pd
import torch
from onehot_class import onehot_1
from ordinal_class import label_1
from sklearn.preprocessing import OrdinalEncoder
import pickle

st.title("Predicting Depression")

with st.form(key="data_form"):

 Age = st.number_input("Enter Your age",min_value=0, max_value=100, value=0,key="Age")
 City = st.text_input("Enter your City")
 Working_Professional_or_Student = st.selectbox("select Working_Professional_or_Student" , [' ','Student','Working Professional'])
 Gender = st.selectbox("Enter your Gender", [' ','Female', 'Male'])
 Profession = st.text_input("Enter your Profession")
 Academic_Pressure = st.number_input("Rate Academic_Pressure(0-5)", min_value=0, max_value=5, value=0,key="Academic_Pressure")
 Work_Pressure = st.number_input("Rate your number Work_Pressure(0-5)", min_value=0, max_value=5, value=0,key = "Work_Pressure")
 Study_Satisfaction = st.number_input("Rate your number Study_Satisfaction(0-5)", min_value=0, max_value=5, value=0,key = "Study_Satisfaction")
 Job_Satisfaction = st.number_input("Rate your number Job_Satisfaction(0-5)", min_value=0, max_value=5, value=0,key="Job_Satisfaction")
 Sleep_Duration = st.number_input("Rate your number Sleep_Duration(0-12)", min_value=0, max_value=10, value=0,key="Sleep_Duration")
 Work_Study_Hours = st.number_input("Rate your number Work_Study_Hours(0-12)", min_value=0, max_value=12, value=0,key="Work_Study_Hours")
 Dietary_Habits = st.selectbox("Select your Dietary status",['  ','Moderate','Unhealthy','Healthy'])
 Have_you_ever_had_suicidal_thoughts = st.selectbox("Select if Have you ever had suicidal thoughts ?" , ['  ','Yes','No'])
 Financial_Stress = st.number_input("Rate your Financial stress (0-5)", min_value=0, max_value=5, value=0,key="Financial_Stress")
 Family_History_of_Mental_Illness = st.selectbox("Select Family History of Mental Illness" , ['  ','Yes','No'])
 submitted = st.form_submit_button("Submit")

if submitted:
    data = {"Age" : [Age],"City":[City],"Working Professional or Student":[Working_Professional_or_Student],"Profession":[Profession],"Academic Pressure":[Academic_Pressure],"Gender":[Gender],"Work Pressure":[Work_Pressure],"Study Satisfaction":[Study_Satisfaction],"Job Satisfaction":[Job_Satisfaction],"Sleep Duration":[Sleep_Duration],"Dietary Habits":[Dietary_Habits],"Have you ever had suicidal thoughts ?" :[Have_you_ever_had_suicidal_thoughts],"Work/Study Hours" : [Work_Study_Hours],"Financial Stress":[Financial_Stress],"Family History of Mental Illness":[Family_History_of_Mental_Illness]}
    data_1 = pd.DataFrame(data)
    data_1.replace('', pd.NA, inplace=True)
    if data_1.isnull().sum().sum() != 0:
      st.warning("Kindly fill all the details")
      empty_columns = data_1.columns[data_1.isnull().all()]
      st.write("Empty columns : ", empty_columns[0])
    else:  
     st.success("Loading ...............")
    
    
    
    cat88 = ['City','Working Professional or Student','Profession']
    cat99 = ['Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                  'Family History of Mental Illness']
    
    hot_11 = onehot_1(data_1,cat99)
    one_dfx = hot_11.hot()

    label_20 = label_1(one_dfx,cat88)
    label_dfx = label_20.lab()

    if 'Gender_Male' not in label_dfx.columns:
      label_dfx['Gender_Male'] = 0
    if 'Gender_Female' not in label_dfx.columns:
      label_dfx['Gender_Female'] = 0
    if 'Dietary Habits_Healthy' not in label_dfx.columns:
      label_dfx['Dietary Habits_Healthy'] = 0
    if 'Dietary Habits_Moderate' not in label_dfx.columns:
      label_dfx['Dietary Habits_Moderate'] = 0
    if 'Dietary Habits_Unhealthy' not in label_dfx.columns:
      label_dfx['Dietary Habits_Unhealthy'] = 0
    if 'Have you ever had suicidal thoughts ?_No' not in label_dfx.columns:
      label_dfx['Have you ever had suicidal thoughts ?_No'] = 0
    if 'Have you ever had suicidal thoughts ?_Yes' not in label_dfx.columns:
      label_dfx['Have you ever had suicidal thoughts ?_Yes'] = 0
    if 'Family History of Mental Illness_No' not in label_dfx.columns:
      label_dfx['Family History of Mental Illness_No'] = 0
    if 'Family History of Mental Illness_Yes' not in label_dfx.columns:
      label_dfx['Family History of Mental Illness_Yes'] = 0
    


    colum = ['City_9']

    encoder_o = OrdinalEncoder()
    encoder_n = encoder_o.fit_transform(data_1[['City']])
    encoder_d = encoder_o.inverse_transform(encoder_n)
    df_new = pd.DataFrame(encoder_d, columns=colum)

    x_test_pred = torch.tensor(label_dfx.values,dtype = torch.float32)
    
    with open("C:/Users/Hp/OneDrive/Desktop/Depressionmodel/Depressionmodel1.pkl", "rb") as f:
        loaded_model = pickle.load(f)


    loaded_model.eval()
    with torch.no_grad():
          c_pred = loaded_model(x_test_pred)
          c_pred_labels = (c_pred > 0.5).int().view(-1)
    label_dfx['Depression'] = c_pred_labels.numpy()
    st.write(label_dfx['Depression'])
    

    