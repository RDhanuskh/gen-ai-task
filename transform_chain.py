from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.3
)

# ---------------- TRANSFORM CHAIN ----------------
# This cleans the text before AI sees it
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("!", "")
    text = text.replace("@", "")
    return text.strip()

transform_chain = RunnableLambda(clean_text)

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
    "Summarize this text in one line:\n{text}"
)

llm_chain = prompt | llm

# ---------------- FULL PIPELINE ----------------
full_chain = transform_chain | (lambda x: {"text": x}) | llm_chain

# ---------------- RUN ----------------
messy_text = """
Summarize this text in one line:
machine learning  it is very    powerful

"""

result = full_chain.invoke(messy_text)

print("\nFinal Output:\n")
print(result.content)
