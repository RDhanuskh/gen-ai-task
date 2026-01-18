from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ GOOGLE_API_KEY not found in .env file")
    exit()

# Use NEW Gemini model (IMPORTANT)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words."
)

# New LangChain syntax (replaces LLMChain)
chain = prompt | llm

# Run chain
response = chain.invoke({"topic": "Machine Learning"})

print("\nOutput:\n")
print(response.content)
