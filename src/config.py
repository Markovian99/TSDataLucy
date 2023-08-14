# Define some dummy data
MODELS = ["OpenAI: gpt-3.5-turbo", "OpenAI: gpt-3.5-turbo-16k", "OpenAI: gpt-4", "OpenAI: gpt-4-32k", "Google: Bard", "GPT-J-7B"]

EMBEDDING_MODELS = ["all-MiniLM-L6-v2"]

TEMPERATURE = 0
MAX_TOKENS = 8193
DATA_FRACTION=.8

DATE_VAR = "date"
APP_NAME = "Time Series Data Analyizer - Lucy"

# Aggregation defaults to sum unless feature is specified in list here
MEAN_AGG = []

# make sure to include the trailing slash
PROCESSED_DOCUMENTS_DIR = "../data/processed/"
REPORTS_DOCUMENTS_DIR = "../data/reports/"
