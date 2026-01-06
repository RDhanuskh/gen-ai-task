from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool

# ---------------- TOOL ----------------
@tool
def calculator(expression: str) -> str:
    """Solve mathematical expressions"""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations"
    )
]

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",   # ✅ Use this model
    temperature=0
)

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
    """
You are an intelligent agent.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use this format strictly:

Question: {input}
Thought: reasoning
Action: one of [{tool_names}]
Action Input: input
Observation: result
Thought: final reasoning
Final Answer: answer

Begin!

Question: {input}
{agent_scratchpad}
"""
)

# ---------------- AGENT ----------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ---------------- RUN ----------------
query = "Calculate 25 * 15 and also 12 * 4"
response = agent_executor.invoke({"input": query})

print("\nFinal Answer:")
print(response["output"])
