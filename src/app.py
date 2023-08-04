import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATE_VAR, DATA_FRACTION, APP_NAME, MEAN_AGG
from app_utils import generate_responses, initialize_session_state, identify_categorical, process_ts_data, num_tokens_from_string, identify_features_to_analyze

# default session state variables
initialize_session_state()

# App layout
st.title(APP_NAME)

#general context for prompts
general_context = "" 
with st.sidebar:
    model = st.selectbox(f"Select Model ", MODELS)
    st.session_state["generation_model"]=model
    general_context = st.text_area(f"Enter extra prompt details (added to the beginning of all prompts)", "You are a data analyst with a strong business intuition.")
general_context=general_context+" "

# User prompt
brief_description = st.text_input("Please provide a brief description of the data file (e.g. This is market data for the S&P500)", "")
if len(brief_description)>0:
        general_context = general_context + "The following brief description of the data was provided: "+ brief_description + "\n"

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    #copy the file to "raw" folder
    with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())

    st.session_state["uploaded_file"] = uploaded_file.name
    dataframe = pd.read_csv(uploaded_file)
    dataframe[DATE_VAR] = pd.to_datetime(dataframe[DATE_VAR])
    st.session_state["categorical_features"]=["None"]+identify_categorical(dataframe)
    if st.session_state["numeric_features"]==[]:
        st.session_state["numeric_features"]=identify_features_to_analyze(dataframe)
        try:
            # use genai to create the default list of features to analyze
            st.session_state["features_to_analyze"]=identify_features_to_analyze(dataframe,use_llm=True,prompt_prefix=general_context)
            print("Features to analyze: ")
            print(st.session_state["features_to_analyze"])
        except Exception as e:
            print(e)
            st.session_state["features_to_analyze"]=st.session_state["numeric_features"]
    st.session_state["start_date"]=dataframe[DATE_VAR].min()
    st.session_state["end_date"]=dataframe[DATE_VAR].max()
    st.write(dataframe)

st.header("Summary Analysis Reports")
col1, col2 = st.columns(2)
with col1:
    # checkbox for each type of report
    field_summary = st.checkbox("Field Descriptions", value=True)
    data_summary = st.checkbox("Data Summary", value=True)
with col2:
    # checkbox for each type of report
    recent_summary = st.checkbox("Recent Data Analysis", value=True)
    trend_summary = st.checkbox("Trend Summary", value=False)

if uploaded_file is not None:    
    col1, col2 = st.columns(2)
    with col1:
        d_min = st.date_input("Analysis Start Date", value=st.session_state["start_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])
    with col2:
        d_max = st.date_input("Analysis End Date", value=st.session_state["end_date"], min_value=st.session_state["start_date"], max_value=st.session_state["end_date"])

    selected_features = st.multiselect('What are the metrics / features to analyze?', st.session_state["numeric_features"], default=st.session_state["features_to_analyze"])
    drop_features = [f for f in st.session_state["numeric_features"] if f not in selected_features and f!=DATE_VAR]

    # streamlit header
    st.header("Group By Analysis")
    by_var = st.selectbox(f"Select Group By Variable ", st.session_state["categorical_features"])

    if by_var != "None":
        col1, col2 = st.columns(2)
        with col1:
            # checkbox for each type of report
            data_summary_by_group = st.checkbox("Data Summary by Group", value=True)
            compare_by_group = st.checkbox("Compare Data by Group", value=False)            
        with col2:
            # checkbox for each type of report
            recent_summary_by_group = st.checkbox("Recent Data Analysis by Group", value=False)
            trend_summary_by_group = st.checkbox("Trend Summary by Group", value=False)
            

    
#don't have it appear until responses are generated
clear_button = None

# User prompt
requested_prompt = st.text_input("(Optional) Enter your prompt", "")

# Generate button
generate_button = st.button("Generate Responses")

if generate_button and st.session_state['uploaded_file']:
    template=""
    # if by_var is not set, set it to None
    if by_var =="None":
        by_var = None

    dataframe = pd.read_csv(os.path.join("../data/raw/"+st.session_state["uploaded_file"]))
    dataframe[DATE_VAR] = pd.to_datetime(dataframe[DATE_VAR])
    #subset dataframe based on min and max dates
    dataframe = dataframe[(dataframe[DATE_VAR].dt.date>=d_min) & (dataframe[DATE_VAR].dt.date<=d_max)]
    dataframe = dataframe.drop(drop_features, axis=1)

    # process time series data to save descriptive information for prompts
    process_ts_data(dataframe, DATE_VAR, by_var)

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
        prompt_context = general_context + "Compare the aggregated data from the start period with the most recent period to provide analysis of the most recent period.\n Start period:\n"+ json_start+"\n Recent period:\n"+json_recent
        recent_summary_response = generate_responses(prompt_context, model, template)
        st.header(f"Recent Data Analysis")
        st.write(recent_summary_response)

    if by_var:
        if data_summary_by_group:
            # read in the summary data into a string
            with open('../data/processed/comparison.txt', 'r') as f:
                group_summaries = f.read()           
            prompt_context = general_context + group_summaries + "Please summarize the data provided by group."
            data_summary_response = generate_responses(prompt_context, model, template)
            st.header(f"Data Summary by {by_var}")
            st.write(data_summary_response)
        if compare_by_group:
            # read in the comparison data into a string
            with open('../data/processed/comparison.txt', 'r') as f:
                group_summaries = f.read()        
            prompt_context = general_context + group_summaries + "Please compare the metrics from the different sub-groups to each other."
            comparison_response = generate_responses(prompt_context, model, template)
            st.header(f"{by_var} Comparison Analysis")
            st.write(comparison_response)
        if recent_summary_by_group:
            # read in the data into a strings
            with open('../data/processed/start_by_group.json', 'r') as f:
                json_start_by_group = f.read() 
            with open('../data/processed/recent_by_group.json', 'r') as f:
                json_recent_by_group = f.read()     
            prompt_context = general_context + f"Compare the data from the start period with the most recent period for each {by_var} to provide analysis of the most recent period.\n Start period:\n"+ \
                            json_start_by_group+"\n Recent period:\n"+json_recent_by_group
            recent_summary_response = generate_responses(prompt_context, model, template)
            st.header(f"Recent Data Analysis by {by_var}")
            st.write(recent_summary_response)   

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
