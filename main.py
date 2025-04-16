import pandas as pd
import streamlit as st
import json
import os
import time
from datetime import datetime
from model1 import user_input


#setting page background
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        background: linear-gradient(135deg, #e0f7fa, #80cbc4);
        color: #00332d;
        font-family: 'Segoe UI', sans-serif;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3, h4 {
        color: #004d40;
    }
    .stTextInput > div > div > input {
        background-color: #ffffffcc;
        color: #00332d;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        border: none;
    }
    .stButton > button {
        background-color: #00796b;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #004d40;
        transform: scale(1.02);
    }
    .stSidebar {
        background-color: #004d40 !important;
    }
    .stSidebar h2 {
        color: white !important;
    }
    .stSidebar .css-1v0mbdj {
        color: white !important;
    }
    .stSidebar .stMarkdown {
        color: white !important;
    }
    .stDataFrame {
        background-color: #ffffff33;
    }
    </style>
""", unsafe_allow_html=True)

# JSON log file
LOG_FILE = 'user_logs.json'

# Load existing logs or initialize
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as file:
        logs = json.load(file)
else:
    logs = []

#Setting Headers
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #004d40;'>Enter a news headline or body text to analyze.</p>", unsafe_allow_html=True)

#Getting text from user
text= st.text_input("")

#Main loop of the model
if st.button("Analyze",key="1"):
    if not text:
        st.warning("⚠️ Please enter news text before analyzing")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(4)
        if user_input(text)==0:
            result="Fake"
            st.markdown("<div style='background-color:#FFDDDD;padding:10px;border-radius:5px;'>❌ This news appears to be <b>Fake</b></div>", unsafe_allow_html=True)
        elif user_input(text)==1:
            result="Real"
            st.markdown("<div style='background-color:#DDFFDD;padding:10px;border-radius:5px;'>🟢 This news appears to be <b>Real</b></div>", unsafe_allow_html=True)
            # Log the result
        entry = {
            "timestamp": datetime.now().isoformat(),
             "News": text,
            "prediction":result
         }
        logs.append(entry)
        with open(LOG_FILE, 'w') as file:
            json.dump(logs, file, indent=4)
             

#For Viewing Detection History
if st.checkbox("View History"):
    with open("user_logs.json", "r") as file:
        logs = json.load(file)
    df = pd.DataFrame(logs)
    st.dataframe(df)

        
# ---------- Sidebar ----------
st.sidebar.markdown("""
    <div style='
        background-color: #00695c;
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
    '>
        <h3>ℹ️ About This Project</h3>
        <p>This model was developed as part of the <b>CS196 Final Project</b>.</p>
        <ul>
            <li>It detects whether a news input is <i>Real</i> or <i>Fake</i>.</li>
            <li>⚠️ The model is not 100% accurate.</li>
            <li>Based on limited training data.</li>
            <li>Use for <b>informational purposes</b> only.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)



    

