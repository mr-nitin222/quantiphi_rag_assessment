'''
app.py: An application to run the streamlit app for bio_rag.py

This code uses saved embeddings of chunks taken from textbook

created by : Nitin Mishra
created date: 15th October 2024
'''

import streamlit as st
import subprocess
import bio_rag

# Putting a title to our app
st.title("BIO RAG : An interactive platform to deep dive into elementary biology")

# Asking the user to put up a question
user_query = st.chat_input("Say something")
if user_query:
    st.write(f"Question: {user_query}")
    # subprocess.run(["python", "script.py"])
    st.write(f"Answer:\n{bio_rag.main(user_query)}")
    