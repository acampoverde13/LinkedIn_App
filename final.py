import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

st.markdown("# This is my app!")
st.markdown("Please enter values for all of the user features below:")
#st.markdown("Use this information below for income & education levels:)

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

income = st.number_input("What is their income level (1-9)", min_value=1, max_value=9,step=1, placeholder="")
education = st.number_input("What is their education level (1-8)", min_value=1, max_value=8,step=1, placeholder="")
parent = st.number_input("Are they a parent? (1-Yes,0-No)", min_value=0, max_value=1,step=1, placeholder="")
married = st.number_input("Are they married? (1-Yes,0-No)", min_value=0, max_value=1,step=1, placeholder="")
gender = st.number_input("What is their gender? (1-Male,0-Female)", min_value=0, max_value=1,step=1, placeholder="")
age = st.number_input("What is their age?",step=1, placeholder="")

ui_person.extend([income, education, parent, married, gender, age])

predicted_class = lr.predict([ui_person])
probs = lr.predict_proba([ui_person])

st.write(f"Predicted class: {predicted_class[0]}") # 0=not LinkedIn user, 1=LinkedIn user
st.write(f"Probability that this person is a LinkedIn user: {probs[0][1]}")