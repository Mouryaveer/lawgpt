import streamlit as st
import main

st.title("LawGPT-Mk1")

query = st.text_input("Enter your query related to legal domain:", key="query")

if query:
    with st.spinner("Generating response..."):
        response = main.ragu(query)
        st.write("Response:")
        st.write(response)
        st.balloons()    
