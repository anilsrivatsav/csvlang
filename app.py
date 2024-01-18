# type: ignore
import streamlit as st
from utils import get_answer_csv
from utils import get_conversational_chain

st.header("Chat with multiple CSV using Ai")

# Correcting indentation here
model_choice = st.selectbox("Choose the AI Model:", ["OpenAI", "Gemini Pro"])
with st.sidebar:
    st.title("Menu:")
  
    st.title("Upload CSV files:")
    uploaded_file = st.file_uploader("Upload a csv file", type=["csv"], accept_multiple_files=True)


if uploaded_file is not None:
    query = st.text_area("Ask any question related to the document")
    button = st.button("Submit")
    if button:
        st.write(get_answer_csv(uploaded_file, query))