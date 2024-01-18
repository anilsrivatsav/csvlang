# type: ignore
import os
from typing import TextIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import openai
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI

#openai.api_key = st.secrets["OPENAI_API_KEY"]
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_answer_csv(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")

    # Create an agent using OpenAI and the Pandas dataframe
    agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False)
    #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)

    # Run the agent on the given query and return the answer
    #query = "whats the square root of the average age?"
    answer = agent.run(query)
    return answer

  


def get_conversational_chain(model_choice, conversation_history):
    
    formatted_history = "\n".join(conversation_history)
    
    prompt_template = """
    
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    {conversation_history}
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    if model_choice == "OpenAI":
        model = OpenAI()
    elif model_choice == "Gemini Pro":
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain