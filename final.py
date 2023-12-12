import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

st.markdown("## Welcome to the LinkedIn User Prediction Tool!")
st.markdown("### Please enter values for all of the user features below:")
st.write("""
         Use this information below for income & education levels:\n
         For income:\n
            1.  Less than $10,000\n
            2.  10 to under $20,000\n
            3.  20 to under $30,000\n           
            4.  30 to under $40,000\n
            5.  40 to under $50,000\n           
            6.  50 to under $75,000\n
            7.  75 to under $100,000\n          
            8.  100 to under $150,000\n
            9.  $150,000 or more\n
         For education:\n
            1.  Less than high school (Grades 1-8 or no formal schooling)\n
            2.  High school incomplete (Grades 9-11 or Grade 12 with NO diploma)\n
            3.  High school graduate (Grade 12 with diploma or GED certificate)\n
            4.  Some college, no degree (includes some community college)\n
            5.  Two-year associate degree from a college or university\n
            6.  Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)\n
            7.  Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)\n
            8.  Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)\n
         """)

def clean_sm(x):
    return np.where(x == 1, 1, 0)

s = pd.read_csv('social_media_usage2.csv')
ss = s[["income","educ2","par","marital","gender","age"]].copy()
ss = ss.rename(columns={"educ2": "education","par":"parent","marital":"married"})
ss["sm_li"]= s["web1h"].apply(clean_sm)
ss = ss.loc[(ss["income"] < 10) & (ss["education"] < 9) & (ss["age"]<99)]
ss[["gender", "married","parent"]] = ss[["gender","married","parent"]].apply(clean_sm)


y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","gender","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,    
                                                    random_state=987)
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
ui_person = []

income = st.number_input("What is their income level (1-9)",min_value=1, max_value=9,step=1, placeholder="")
education = st.number_input("What is their education level (1-8)",min_value=1, max_value=8,step=1, placeholder="-")
parent = st.number_input("Are they a parent? (1-Yes,0-No)", min_value=0,max_value=1,step=1, placeholder="-")
married = st.number_input("Are they married? (1-Yes,0-No)", min_value=0,max_value=1,step=1, placeholder="-")
gender = st.number_input("What is their gender? (1-Male,0-Female)", min_value=0,max_value=1,step=1, placeholder="-")
age = st.number_input("What is their age?",step=1, placeholder="")

ui_person.extend([income, education, parent, married, gender, age])


predicted_class = lr.predict([ui_person])
probs = lr.predict_proba([ui_person])
if predicted_class == 1:
    st.write("#### This person is likely a LinkedIn user")
else:
    st.write("#### This person is likely not a LinkedIn user")
    
st.write(f"#### The probability that this person is a LinkedIn user is: {probs[0][1]}")