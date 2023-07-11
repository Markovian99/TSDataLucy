import pandas as pd
import streamlit as st
import os
import json


from config import MODELS, KNOWLEDGE_BASES, K, FETCH_K, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, MAX_TOKENS

from bardapi import Bard
import openai

# make sure load_dotenv is run from main app file first
openai.api_key = os.getenv('OPENAI_API_KEY')
bard = Bard(token=os.getenv('BARD_API_KEY'))

def initialize_session_state():
    # Initialise all session state variables with defaults
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "responses": [],
        "uploaded_file": None,
        "generation_model": None,
        "generation_prompt": None,
        "generation_prompt_final": None,
        "k": K,
        "fetch_k": FETCH_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# This is a dummy function to simulate generating responses.
def generate_responses(prompt, model, template, knowledge_base, context):
    response = "No model selected"

    if model != "None":
        st.session_state["generation_models"] = model
        st.session_state["generation_prompt"] = prompt

        #### ADD KNOWLEDGE BASE LOOKUP
        final_prompt=prompt
        st.session_state["generation_prompt_final"] = final_prompt

        if model.startswith("Google"):
            this_answer = bard.get_answer(final_prompt)
            response = this_answer['content']
        elif model.startswith("OpenAI: "):
            response_full = openai.ChatCompletion.create( model=model[8:],   messages=[{"role": "user", "content": final_prompt }])
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



def process_ts_data(df, date_var='date'):
    # Convert 'date' column to datetime type
    df[date_var] = pd.to_datetime(df[date_var])

    # Set 'date' as the index of the dataframe
    df.set_index(date_var, inplace=True)

    # Summarize the dataframe
    st.write(df.describe())
    data = json.loads(df.describe().to_json())
    # Now, let's write this data to a file
    with open('../data/processed/summary.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # Summarize the dataframe
    st.write(df.tail(14))
    data = json.loads(df.tail(14).to_json())
    # Now, let's write this data to a file
    with open('../data/processed/last_2_weeks.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)






# # Define all containers upfront to ensure app UI consistency
# # container to upload files
# upload_container = st.container()
# # container to enter any datasource string
# datasource_container = st.container()
# # container to display infos stored to session state
# # as it needs to be accessed from submodules
# st.session_state["info_container"] = st.container()
# # container for chat history
# response_container = st.container()
# # container for text box
# text_container = st.container()
