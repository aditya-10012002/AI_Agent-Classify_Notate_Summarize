import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
# llm = ChatOpenAI(model="gpt-3.5-turbo")

response = llm.invoke("Hello, are you working?")

print(response.content)