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
description = st.text_input("Please provide a brief description of the data file (e.g. This is market data for the S&P500)", "")

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

    prompt_context=""

    if st.session_state['uploaded_file'] :
        # Open the file in read mode
        with open('../data/processed/head.json', 'r') as json_file:
            data = json.load(json_file)
        # Convert the Python dictionary back to a JSON string
        head_string = json.dumps(data, indent=4)

        # Open the file in read mode
        with open('../data/processed/summary.json', 'r') as json_file:
            data = json.load(json_file)
        # Convert the Python dictionary back to a JSON string
        json_string = json.dumps(data, indent=4)
        prompt_context = prompt_context + "\nPlease consider the following json string summarizing the data in your response: \n"+ json_string 
        
        # Open the file in read mode
        with open('../data/processed/tail.json', 'r') as json_file:
            data = json.load(json_file)
        # Convert the Python dictionary back to a JSON string
        json_string2 = json.dumps(data, indent=4)
        prompt_context = prompt_context+ "\nPlease consider the following json string containing the most recent data \n" + json_string2 

    # Print the JSON string
    prompt_requested = prompt + prompt_context + "\n" + prompt

    print(f"Length of prompt: {len(prompt)}")

    prompt_description = description + "\n This is an example of the first set of rows \n"+head_string +"\n"+"Please decribe what this data may represent."
    description_response = generate_responses(prompt_description, model, template, knowledge_base, context)
    requested_response = generate_responses(prompt_requested, model, template, knowledge_base, context)

    st.session_state["generated_responses"] = True
    st.header(f"Description of what this data may represent")
    st.write(description_response)
    st.header(prompt)
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

    st.write(st.session_state["responses"])