import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 1. List available models
for model in client.models.list():
    print(model.name)

# 2. Test generation
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello! Summarize what you can do.",
)
print(response.text)