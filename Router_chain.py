from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0
)

# ---------------- MATH CHAIN ----------------
math_prompt = PromptTemplate.from_template(
    "Solve this math problem:\n{input}"
)
math_chain = math_prompt | llm

# ---------------- TEXT CHAIN ----------------
text_prompt = PromptTemplate.from_template(
    "Answer this in simple words:\n{input}"
)
text_chain = text_prompt | llm

# ---------------- ROUTER PROMPT ----------------
router_prompt = PromptTemplate.from_template(
    """
Decide if the input is a math problem or a general question.

If it is math, reply: math
If it is general text, reply: text

Input: {input}
Answer:
"""
)

router_chain = router_prompt | llm | RunnableLambda(lambda x: x.content.strip().lower())

# ---------------- ROUTER LOGIC ----------------
router = RunnableBranch(
    (lambda x: "math" in x["route"], math_chain),
    (lambda x: "text" in x["route"], text_chain),
    text_chain   # default
)

# ---------------- FULL PIPELINE ----------------
full_chain = {
    "route": router_chain,
    "input": lambda x: x
} | router

# ---------------- RUN ----------------
print("\n--- Math Example ---")
print(full_chain.invoke("25 * 4 + 10").content)

print("\n--- Text Example ---")
print(full_chain.invoke("What is Artificial Intelligence?").content)
