import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATE_VAR, DATA_FRACTION, APP_NAME
from app_utils import generate_responses, initialize_session_state, identify_categorical, process_ts_data, num_tokens_from_string, identify_features_to_analyze
from app_sections import run_upload_and_settings, run_report_gererator, run_chatbot

# default session state variables
initialize_session_state()

# App layout
st.title(APP_NAME)

#general context for prompts
with st.sidebar:
    model = st.selectbox(f"Select Model ", MODELS)
    app_task = st.radio(f"Select Task ", ["Report Generator", "Chatbot"])
    st.session_state["generation_model"]=model
    st.session_state["general_context"] = st.text_area(f"Enter extra prompt details (added to the beginning of all prompts)", "You are a data analyst with a strong business intuition.")+" "


#don't have it appear until responses are generated
clear_button = None

# start with the upload and settings 
run_upload_and_settings()

if app_task == "Report Generator" and st.session_state['uploaded_file']:
    run_report_gererator()

elif app_task == "Chatbot" and st.session_state['uploaded_file']:
    run_chatbot()

