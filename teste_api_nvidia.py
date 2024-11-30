from openai import OpenAI
import os
from dotenv import dotenv_values

config = dotenv_values(".env")
OPENAI_API_KEY = config["OPENAI_API_KEY"].strip()



OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
TEMPERATURE = float(os.getenv('TEMPERATURE'))
TOP_P = float(os.getenv('TOP_P'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = OPENAI_API_KEY
)

completion = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-70b-instruct",
  messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.5,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

