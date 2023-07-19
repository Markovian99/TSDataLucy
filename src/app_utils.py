import pandas as pd
import streamlit as st
import os
import json
import tiktoken
import numpy as np


from config import MODELS, TEMPERATURE, MAX_TOKENS, DATA_FRACTION

from bardapi import Bard
import openai


# make sure load_dotenv is run from main app file first
openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

bard = Bard(token=os.getenv('BARD_API_KEY'))


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def identify_categorical(df, unique_threshold=100, max_portion=0.1):
    """Identify categorical columns in a dataframe."""
    categorical_cols = []
    max_unique = min(int(len(df) * max_portion), unique_threshold)
    
    potential_categorical_cols = df.select_dtypes(include=['int64', 'object']).columns
    for col in potential_categorical_cols:
        if df[col].nunique() <= max_unique and df[col].nunique() > 1:
            categorical_cols.append(col)
            
    return categorical_cols

def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "responses": [],
        "categorical_features": [],
        "start_date": None,
        "end_date": None,
        "uploaded_file": None,
        "generation_model": None,
        "generation_prompt": None,
        "generation_prompt_final": None,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# This is a dummy function to simulate generating responses.
def generate_responses(prompt, model, template, temperature=0):
    response = "No model selected"

    if model != "None":
        st.session_state["generation_models"] = model
        st.session_state["generation_prompt"] = prompt

        final_prompt=prompt
        st.session_state["generation_prompt_final"] = final_prompt

        if model.startswith("Google"):
            this_answer = bard.get_answer(final_prompt)
            response = this_answer['content']
        elif model.startswith("OpenAI: "):
            response_full = openai.ChatCompletion.create( model=model[8:],   messages=[{"role": "user", "content": final_prompt }], temperature=temperature)
            response = response_full['choices'][0]['message']['content']

        st.session_state["responses"].append(response)

    return response


# This is a dummy function to simulate saving responses to a database.
def save_to_db(position):#(model, prompt, final_prompt, response, prompt_template="", extra_context="", rating=-1):
    today = date.today()
    prompt_template=""
    extra_context=""
    df = pd.DataFrame([[today,st.session_state["generation_model"], st.session_state["generation_prompt"], prompt_template, extra_context,
                       st.session_state["generation_prompt_final"], st.session_state["responses"], st.session_state["ratings"]]], 
                       columns=['date', 'model','prompt','prompt_template','extra_context', 'final_prompt', 'response', 'rating'])
    #st.write(f"Saving response: {response}")

def process_ts_data(df_input:pd.DataFrame, date_var='date', by_var=None):
    """Process time series data and save to various json files for streamlit app.
    - summary_all.json: summary of all columns
    - summary.json: summary of numeric columns only
    - head.json: first 7 rows of dataframe
    - start.json: first 7 days of dataframe
    - recent.json: last 7 days of dataframe
    - compare.json: numeric summary comparison by group    
    """
    df = df_input.copy()

    # Summarize the dataframe (all and numeric only)
    data = json.loads(df.describe(include='all').to_json())
    # Now, let's write this data to a file
    with open('../data/processed/summary_all.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    data = json.loads(df.describe().to_json())
    # Now, let's write this data to a file
    with open('../data/processed/summary.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    use_rows=7
    if len(df.columns)>10:
        use_rows=2
    # Write the head of dataframe to a file
    data = json.loads(df.head(use_rows).to_json())
    # Now, let's write this data to a file
    with open('../data/processed/head.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # subset to all numeric columns and date column and by_var for analysis
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols = df.select_dtypes(include=numerics).columns.to_list() + [date_var]
    if by_var: cols.append(by_var)
    df = df[cols]

    # if we need to only keep a subset of by_var values due to size of data
    by_var_values = []

    # keep only the first 7 days in the dataframe and write json file
    data = df[df[date_var] < df[date_var].min() + pd.Timedelta(days=7)]
    #check if by_var is not None and if by_var, date_var combination is unique
    if by_var and len(data[[by_var, date_var]].groupby([by_var, date_var]).size())==len(data):
        #set index as by_var and date_var sorted for json file
        data = data.sort_values([by_var, date_var])
        data.set_index([by_var, date_var], inplace=True, drop=False)
    data_json = data.to_json()
    num_tokens = num_tokens_from_string(data_json)
    if  by_var and num_tokens > int(DATA_FRACTION*MAX_TOKENS):
        # create a subset of the dateframe to a sample of the by_var
        data = data[data[by_var].isin(np.random.choice(data[by_var].unique(),size=int(len(data[by_var].unique())*DATA_FRACTION*MAX_TOKENS/num_tokens),replace=False))]
        #save unique values of by_var
        by_var_values = data[by_var].unique()
        data_json = json.loads(data.to_json())
        # write streamlit warning that the data was subset
        st.warning(f"Start data was subset to {len(data)} rows to fit within the token limit of {MAX_TOKENS} tokens.")
    else: 
        data_json = json.loads(data_json)
    with open('../data/processed/start.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # keep only the last 7 days in the dataframe and write json file
    data = df[df[date_var] > df[date_var].max() - pd.Timedelta(days=7)]  
    #check if by_var is not None and if by_var, date_var combination is unique
    if by_var and len(data[[by_var, date_var]].groupby([by_var, date_var]).size())==len(data):
        #set index as by_var and date_var sorted for json file
        data = data.sort_values([by_var, date_var])
        data.set_index([by_var, date_var], inplace=True, drop=False)
    data_json = data.to_json()
    num_tokens = num_tokens_from_string(data_json)
    if  by_var and num_tokens > int(DATA_FRACTION*MAX_TOKENS):
        if len(by_var_values)>0 and len(data[data[by_var].isin(by_var_values)])>0:
            data = data[data[by_var].isin(by_var_values)]
        else:
            # create a subset of the dateframe to a sample of the by_var
            data = data[data[by_var].isin(np.random.choice(data[by_var].unique(),size=int(len(data[by_var].unique())*DATA_FRACTION*MAX_TOKENS/num_tokens),replace=False))] 
        data_json = json.loads(data.to_json())
        # write streamlit warning that the data was subset
        st.warning(f"Recent data was subset to {len(data)} rows to fit within the token limit of {MAX_TOKENS} tokens.")
    else: 
        data_json = json.loads(data_json)
    with open('../data/processed/recent.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # if by_var is not None, create a summary comparison by group
    if by_var:
        group_counts = df[by_var].value_counts()
        group_summaries = ""
        for group_name in group_counts.index:
            group_summaries = group_summaries + str(group_name)+":\n" + df[df[by_var]==group_name].describe().to_json() + "\n\n"
            # count the length of the prompt
            prompt_len = num_tokens_from_string(group_summaries)
            #if prompt is too long, exit for loop
            if prompt_len>int(DATA_FRACTION*MAX_TOKENS):
                st.warning(f"Comparison data may be subset to fit within the token limit.")
                break
        #now let's write the test to a file
        with open('../data/processed/comparison.txt', 'w') as f:
            f.write(group_summaries)