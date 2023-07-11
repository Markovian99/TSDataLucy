import streamlit as st
from datetime import date
import pandas as pd
from io import StringIO
import json
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

import app_utils

from bardapi import Bard
import openai


from config import MODELS, KNOWLEDGE_BASES, K, FETCH_K, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, MAX_TOKENS, DATE_VAR
from app_utils import generate_responses, initialize_session_state, process_ts_data

 
# # default session state variables
initialize_session_state()

# App layout
st.title("Time Series Data Analyizer - Lucy")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    print(uploaded_file.name)
    print(os. getcwd() )
    #copy the file to "raw" folder
    with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())

    st.session_state["uploaded_file"] = uploaded_file.name

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)

    process_ts_data(dataframe, DATE_VAR)

    st.write(dataframe)

# User prompt
prompt = st.text_input("Enter your prompt", "")

# Generate button
generate_button = st.button("Generate Responses")
    
#don't have it appear until responses are generated
clear_button = None


model = st.selectbox(f"Select Model ", MODELS)
template = st.text_input(f"Enter prompt template ", "")
knowledge_base = st.selectbox(f"Select Knowledge Base ", KNOWLEDGE_BASES)
context = st.text_input(f"Enter context for Knowledge Base ", "")

if generate_button :

    if st.session_state['uploaded_file'] :
        # Open the file in read mode
        with open('../data/processed/summary.json', 'r') as json_file:
            data = json.load(json_file)

        # Convert the Python dictionary back to a JSON string
        json_string = json.dumps(data, indent=4)


        
        # Open the file in read mode
        with open('../data/processed/last_2_weeks.json', 'r') as json_file:
            data = json.load(json_file)

        # Convert the Python dictionary back to a JSON string
        json_string2 = json.dumps(data, indent=4)

        # Print the JSON string
        prompt = prompt + " Please consider the following json string summarizing the data in your response: \n"+ json_string + \
            "\n Please also consider the following json string containing the most recent data \n " + json_string2 + \
            "\n" + prompt

        print(f"Length of prompt: {len(prompt)}")
        
    

    response = generate_responses(prompt, model, template, knowledge_base, context)
    st.session_state["generated_responses"] = True
    # placeholder_list[i] = st.empty()
    st.header(f"Response from Model")
    st.write(response)

if st.session_state["generated_responses"] and not st.session_state["cleared_responses"]:
    clear_button = st.button("Clear Responses")

if clear_button and not st.session_state["cleared_responses"]:        
    print(st.session_state["responses"])
    st.session_state["generated_responses"]=False
    st.session_state["responses"] = []
    st.session_state["cleared_responses"]=True

elif clear_button:
    st.write("No responses to clear - please generate responses")
        # responses = []
        # ratings = [None, None, None]

# Display responses 
if st.session_state["generated_responses"] and not generate_button:
    # placeholder_list[i] = st.empty()
    st.header(f"Response ")

    st.write(st.session_state["responses"][-1])