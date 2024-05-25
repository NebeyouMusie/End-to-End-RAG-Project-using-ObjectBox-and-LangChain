import os
from dotenv import load_dotenv

# load dotenv
def load_config():
    return load_dotenv()

# function to get groq api key
def get_groq_api():
    return os.getenv('GROQ_API_KEY')