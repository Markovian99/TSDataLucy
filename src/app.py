import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATE_VAR
from app_utils import generate_responses, initialize_session_state, identify_categorical, process_ts_data, num_tokens_from_string

# default session state variables
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
    dataframe = pd.read_csv(uploaded_file)
    dataframe[DATE_VAR] = pd.to_datetime(dataframe[DATE_VAR])
    st.session_state["categorical_features"]=["None"]+identify_categorical(dataframe)
    st.session_state["start_date"]=dataframe[DATE_VAR].min()
    st.session_state["end_date"]=dataframe[DATE_VAR].max()
    st.write(dataframe)


col1, col2 = st.columns(2)
with col1:
    # checkbox for each type of report
    field_summary = st.checkbox("Field Descriptions", value=True)
    data_summary = st.checkbox("Data Summary", value=True)
    recent_summary = st.checkbox("Recent Data Analysis", value=True)
with col2:
    # checkbox for each type of report
    trend_summary = st.checkbox("Trend Summary", value=False)
    compare_bygroup = st.checkbox("Compare by Group", value=False)

if uploaded_file is not None:    
    col1, col2 = st.columns(2)
    with col1:
        d_min = st.date_input("Analysis Start Date", value=st.session_state["start_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])
    with col2:
        d_max = st.date_input("Analysis End Date", value=st.session_state["end_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])
    by_var = st.selectbox(f"Select Group By Variable ", st.session_state["categorical_features"])
    
#don't have it appear until responses are generated
clear_button = None


# User prompt
brief_description = st.text_input("Please provide a brief description of the data file (e.g. This is market data for the S&P500)", "")

# User prompt
requested_prompt = st.text_input("(Optional) Enter your prompt", "")

# Generate button
generate_button = st.button("Generate Responses")


model = st.selectbox(f"Select Model ", MODELS)
template = st.text_input(f"Enter extra prompt details (added to end of all prompts)", "")

if generate_button and st.session_state['uploaded_file']:

    # if by_var is not set, set it to None
    if by_var =="None":
        by_var = None

    dataframe = pd.read_csv(os.path.join("../data/raw/"+st.session_state["uploaded_file"]))
    dataframe[DATE_VAR] = pd.to_datetime(dataframe[DATE_VAR])
    #subset dataframe based on min and max dates
    dataframe = dataframe[(dataframe[DATE_VAR].dt.date>=d_min) & (dataframe[DATE_VAR].dt.date<=d_max)]

    # process time series data to save descriptive information for prompts
    process_ts_data(dataframe, DATE_VAR, by_var)

    #general context for prompts
    general_context = "You are a data analyst with a strong business intuition. " 
    if len(brief_description)>0:
        general_context = general_context + "A user provided the following brief description of the data: "+ brief_description + "\n"

    # Open the files in read mode into Python dictionary then back to a JSON string
    with open('../data/processed/head.json', 'r') as json_file:
        data = json.load(json_file)
    json_head = json.dumps(data, indent=4)
    with open('../data/processed/summary_all.json', 'r') as json_file:
        data = json.load(json_file)
    json_summary_all = json.dumps(data, indent=4)
    with open('../data/processed/summary.json', 'r') as json_file:
        data = json.load(json_file)
    json_summary = json.dumps(data, indent=4)
    with open('../data/processed/start.json', 'r') as json_file:
        data = json.load(json_file)
    json_start = json.dumps(data, indent=4)
    with open('../data/processed/recent.json', 'r') as json_file:
        data = json.load(json_file)
    json_recent = json.dumps(data, indent=4)

    prompt_context= general_context + "\n This is an example of the first set of rows \n"+json_head +"\n"+"Please decribe what the data fields may represent."
    #if checked, try to produce a field summary
    if field_summary:
        field_summary_response = generate_responses(prompt_context, model, template)
        st.header(f"Field Summary")
        st.write(field_summary_response)
    
    if data_summary:
        prompt_context = general_context + "Please summarize the data provided and consider this json string summarizing the data: \n"+ json_summary_all
        data_summary_response = generate_responses(prompt_context, model, template)
        st.header(f"Data Summary")
        st.write(data_summary_response)
    if recent_summary:
        prompt_context = general_context + "By comparing the following data summary with the recent data also provided, please provide analysis of the most recent data.\n Summary data:\n"+ json_start+"\n Recent Data:\n"+json_recent
        recent_summary_response = generate_responses(prompt_context, model, template)
        st.header(f"Recent Data Analysis")
        st.write(recent_summary_response)

    if by_var and compare_bygroup:
        dataframe = pd.read_csv(os.path.join("../data/raw/",st.session_state["uploaded_file"]))
        group_counts = dataframe[by_var].value_counts()
        group_summaries = "The following jsons summarize the data in each sub-group to compare.\n\n"
        for group_name in group_counts.index:
            group_summaries = group_summaries + str(group_name)+":\n"
            group_summaries = group_summaries + dataframe[dataframe[by_var]==group_name].describe(include='all').to_json() + "\n\n"
            # count the length of the prompt
            prompt_len = num_tokens_from_string(group_summaries)
            #if prompt is too long, exit for loop
            if prompt_len>int(.9*MAX_TOKENS):
                break

        
        prompt_context = general_context + group_summaries + "Please compare the metrics from the different sub-groups to each other."
        comparison_response = generate_responses(prompt_context, model, template)
        st.header(f"{by_var} Comparison Analysis")
        st.write(comparison_response)

    prompt_context=""
    if len(requested_prompt)>1:
        print(f"Length of prompt: {len(requested_prompt)}")
        # Print the JSON string
        prompt_requested = requested_prompt + prompt_context + "\n" + requested_prompt
        requested_response = generate_responses(prompt_requested, model, template)
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