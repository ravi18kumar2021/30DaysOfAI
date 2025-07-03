import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_FILE = '30DayOfAI.csv'

if Path(DATA_FILE).is_file():
    print(f'{DATA_FILE} is present in the directory')
    df = pd.read_csv(DATA_FILE)
else:
    print(f'{DATA_FILE} is not present in the directory')
    df = pd.DataFrame(columns=["Date", "Topic", "Summary", "Project_Link"])

st.header("📘 30DaysOfAI Learning Tracker App")

st.subheader("📝 Add Today's Progress")

with st.form('form'):
    date = st.date_input("Date", datetime.today())
    topic = st.text_input("Topic", placeholder="eg. Linear Regression")
    summary = st.text_area("What did I learn today?")
    project_link = st.text_input("Project/App Link")
    submitted = st.form_submit_button("Save")
    if submitted:
        new_entry = pd.DataFrame([[date, topic, summary, project_link]], columns=df.columns)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success('Data Saved Successfully!')
    
st.subheader("📊 My Learning Journey")
st.dataframe(df.set_index("Date"))