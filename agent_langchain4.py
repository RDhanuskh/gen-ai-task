from dotenv import load_dotenv
import os
import re

from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool

load_dotenv()

# ----------------------------
# Tools
# ----------------------------
def calculator(text: str) -> str:
    """Evaluates a basic math expression."""
    try:
        allowed = "0123456789+-*/(). "
        if any(c not in allowed for c in text):
            return "Error: Invalid characters"
        return str(eval(text))
    except Exception:
        return "Error: Invalid expression"

def count_words(text: str) -> str:
    """Counts words in the input text."""
    return f"Word count: {len(text.split())}"

def count_sentences(text: str) -> str:
    """Counts sentences in the input text."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return f"Sentence count: {len(sentences)}"

tools = [
    StructuredTool.from_function(calculator),
    StructuredTool.from_function(count_words),
    StructuredTool.from_function(count_sentences),
]

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# ----------------------------
# Manual agent loop
# ----------------------------
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting agent.")
        break

    # Decide which tool to use manually
    if "calculate" in query.lower() or any(c in query for c in "+-*/"):
        response = tools[0].run(query)
    elif "word" in query.lower():
        response = tools[1].run(query)
    elif "sentence" in query.lower():
        response = tools[2].run(query)
    else:
        response = llm.predict(query)

    print("Agent:", response)
