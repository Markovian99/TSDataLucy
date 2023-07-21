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

def mode(x):
    # Check for missing values and empty groups
    if x.isnull().all() or x.empty:
        return np.nan
    else:
        # Use value_counts and return the index of the first occurrence of the maximum value
        mode = x.value_counts().idxmax()
        return mode

def missing(x):
    return x.isnull().sum()

# Specify the aggregation functions for each column
#agg_funcs = {col: [('mode', mode), ('missing', missing)] for col in object_cols}
# Apply groupby and agg
#df_obj = df[object_cols+grouping].groupby(grouping).agg(agg_funcs)


def group_as_needed(df_input, grouping):
    """Group data if needed."""
    df = df_input.copy()
    # only keep numeric and object columns not entirely missing
    numeric_cols =  df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.to_list()    
    numeric_cols = [col for col in numeric_cols if df[col].isnull().sum()<len(df[col]) and col not in grouping]
    object_cols = df.select_dtypes(include=['object']).columns.to_list()
    object_cols = [col for col in object_cols if df[col].isnull().sum()<len(df[col]) and col not in grouping]

    if len(df[grouping].groupby(grouping).size())==len(df):
        #set index as grouping sorted for json file
        df.set_index(grouping, inplace=True, drop=False)
    else:
        # use pandas groupby to aggregate. For objects use the mode and add postfix "mode", for numeric use the sum
        df_num = df[list(set(numeric_cols+grouping))].groupby(grouping).sum()
        if len(object_cols)>0:
            df_obj = df[list(set(object_cols+grouping))].groupby(grouping).agg(lambda x:x.mode())
            df_obj.columns = [f"{col}_mode" for col in df_obj.columns]
            df = pd.concat([df_num, df_obj], axis=1)
        else:
            df = df_num
        df = df.reset_index().set_index(grouping, drop=False)
    return df

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

    # keep only the first 7 days in the dataframe and write json file
    data = df[df[date_var] < df[date_var].min() + pd.Timedelta(days=7)].sort_values([date_var])
    data = group_as_needed(data, grouping=[date_var])
    data_json = json.loads(data.to_json())
    with open('../data/processed/start.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # keep only the last 7 days in the dataframe and write json file
    data = df[df[date_var] > df[date_var].max() - pd.Timedelta(days=7)].sort_values([date_var])
    data = group_as_needed(data, grouping=[date_var])
    data_json = json.loads(data.to_json())
    with open('../data/processed/recent.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # if by_var is not None, create a reports by group
    # if we need to only keep a subset of by_var values due to size of data
    by_var_values = []
    if by_var:
        # keep only the first 7 days in the dataframe and write json file
        data = df[df[date_var] < df[date_var].min() + pd.Timedelta(days=7)].sort_values([by_var, date_var])
        data = group_as_needed(data, grouping=[date_var, by_var])
        data_json = data.to_json()
        num_tokens = num_tokens_from_string(data_json)
        if  num_tokens > int(DATA_FRACTION*MAX_TOKENS/2):
            data = group_as_needed(data.reset_index(drop=True), grouping=[by_var])
            data_json = data.to_json()
            num_tokens = num_tokens_from_string(data_json)
            if  num_tokens > int(DATA_FRACTION*MAX_TOKENS/2):
                # create a subset of the dateframe to a sample of the by_var
                data = data[data[by_var].isin(np.random.choice(data[by_var].unique(),size=int(len(data[by_var].unique())*DATA_FRACTION*MAX_TOKENS/num_tokens),replace=False))]
                #save unique values of by_var
                by_var_values = data[by_var].unique()
                data_json = data.to_json()
                # write streamlit warning that the data was subset
                st.warning(f"Start GroupBy data was subset to {len(data)} rows to fit within the token limit of {MAX_TOKENS} tokens.")
            data_json = json.loads(data_json)        
        else: 
            data_json = json.loads(data_json)
        with open('../data/processed/start_by_group.json', 'w') as json_file:
            json.dump(data_json, json_file, indent=4)

        # keep only the last 7 days in the dataframe and write json file
        data = df[df[date_var] > df[date_var].max() - pd.Timedelta(days=7)]  
        data = group_as_needed(data, grouping=[date_var,by_var])
        data_json = data.to_json()
        num_tokens = num_tokens_from_string(data_json)
        if  num_tokens > int(DATA_FRACTION*MAX_TOKENS/2):
            data = group_as_needed(data.reset_index(drop=True), grouping=[by_var])
            data_json = data.to_json()
            num_tokens = num_tokens_from_string(data_json)
            if  num_tokens > int(DATA_FRACTION*MAX_TOKENS/2):
                if len(by_var_values)>0 and len(data[data[by_var].isin(by_var_values)])>0:
                    data = data[data[by_var].isin(by_var_values)]
                else:
                    # create a subset of the dateframe to a sample of the by_var
                    data = data[data[by_var].isin(np.random.choice(data[by_var].unique(),size=int(len(data[by_var].unique())*DATA_FRACTION*MAX_TOKENS/num_tokens),replace=False))] 
                data_json = json.loads(data.to_json())
                # write streamlit warning that the data was subset
                st.warning(f"Recent GroupBy data was subset to {len(data)} rows to fit within the token limit of {MAX_TOKENS} tokens.")
            data_json = json.loads(data_json) 
        else: 
            data_json = json.loads(data_json)
        with open('../data/processed/recent_by_group.json', 'w') as json_file:
            json.dump(data_json, json_file, indent=4)

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