import streamlit as st
from datetime import date
import pandas as pd
from io import StringIO
import json
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

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
brief_description = st.text_input("Please provide a brief description of the data file (e.g. This is market data for the S&P500)", "")

# User prompt
requested_prompt = st.text_input("(Optional) Enter your prompt", "")

# Generate button
generate_button = st.button("Generate Responses")

# checkbox for each type of report
field_summary = st.checkbox("Field Descriptions", value=False)
data_summary = st.checkbox("Data Summary", value=False)
recent_summary = st.checkbox("Recent Data Analysis", value=False)
trend_summary = st.checkbox("Trend Summary", value=False)

    
#don't have it appear until responses are generated
clear_button = None


model = st.selectbox(f"Select Model ", MODELS)
template = st.text_input(f"Enter prompt template ", "")
knowledge_base = st.selectbox(f"Select Knowledge Base ", KNOWLEDGE_BASES)
context = st.text_input(f"Enter context for Knowledge Base ", "")

if generate_button and st.session_state['uploaded_file']:

    #general context for prompts
    general_context = "You are a data analyst with a strong business intuition. " 
    if len(brief_description)>0:
        general_context = general_context + "A user provided the following brief description of the data: "+ brief_description + "\n"

    # Open the file in read mode into Python dictionary then back to a JSON string
    with open('../data/processed/head.json', 'r') as json_file:
        data = json.load(json_file)
    json_head = json.dumps(data, indent=4)
    prompt_context= general_context + "\n This is an example of the first set of rows \n"+json_head +"\n"+"Please decribe what the data fields may represent."
    #if checked, try to produce a field summary
    if field_summary:
        field_summary_response = generate_responses(prompt_context, model, template, knowledge_base, context)
        st.header(f"Field Summary")
        st.write(field_summary_response)

    # Open the file in read mode into Python dictionary then back to a JSON string
    with open('../data/processed/summary.json', 'r') as json_file:
        data = json.load(json_file)
    json_summary = json.dumps(data, indent=4)
    with open('../data/processed/tail.json', 'r') as json_file:
        data = json.load(json_file)
    json_recent = json.dumps(data, indent=4)
    
    if data_summary:
        prompt_context = general_context + "Please summarize the data provided and consider this json string summarizing the data: \n"+ json_summary
        data_summary_response = generate_responses(prompt_context, model, template, knowledge_base, context)
        st.header(f"Data Summary")
        st.write(data_summary_response)
    if recent_summary:
        prompt_context = general_context + "By comparing the following data summary with the recent data also provided, please provide analysis of the most recent data.\n Summary data:\n"+ json_summary+"\n Recent Data:\n"+json_recent
        recent_summary_response = generate_responses(prompt_context, model, template, knowledge_base, context)
        st.header(f"Recent Data Analysis")
        st.write(recent_summary_response)

    prompt_context=""
    if len(requested_prompt)>1:
        print(f"Length of prompt: {len(requested_prompt)}")
        # Print the JSON string
        prompt_requested = requested_prompt + prompt_context + "\n" + requested_prompt
        requested_response = generate_responses(prompt_requested, model, template, knowledge_base, context)
        st.header(requested_prompt)
        st.write(requested_response)
    
elif generate_button:
    st.write("Please upload a file first")

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