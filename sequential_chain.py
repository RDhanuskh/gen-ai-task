from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ API Key missing")
    exit()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)

# -------- Chain 1: Get Definition --------
definition_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give a detailed definition of {topic}."
)

chain1 = definition_prompt | llm

# -------- Chain 2: Simplify Explanation --------
simplify_prompt = PromptTemplate(
    input_variables=["text"],
    template="Simplify the following text into very easy words:\n{text}"
)

chain2 = simplify_prompt | llm

# -------- Run Sequential Chain --------
topic = "Artificial Intelligence"

# Run Chain 1
definition = chain1.invoke({"topic": topic}).content

# Run Chain 2 using Chain 1 output
simple_explanation = chain2.invoke({"text": definition}).content

print("\n🔹 Original Definition:\n")
print(definition)

print("\n🔹 Simplified Explanation:\n")
print(simple_explanation)
