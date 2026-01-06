# ---------------- IMPORTS ----------------
from langchain.prompts import PromptTemplate
from langchain.llms.fake import FakeListLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# ---------------- TOOL ----------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result as a string.
    Example: "(50 / 5) + (8 * 6)" -> "58"
    """
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

# ---------------- FAKE LLM ----------------
llm = FakeListLLM(
    responses=[
        "Thought: I should calculate the expression.\n"
        "Action: Calculator\n"
        "Action Input: (50 / 5) + (8 * 6)",

        "Observation: 58\n"
        "Final Answer: 58"
    ]
)

# ---------------- PROMPT ----------------
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are an intelligent agent.

You can use the following tools:
{tools}

Tool names: {tool_names}

Follow this format strictly:

Thought:
Action:
Action Input:
Observation:
Final Answer:

Question: {input}

{agent_scratchpad}
"""
)

# ---------------- AGENT ----------------
agent = initialize_agent(
    tools=[calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Updated for latest LangChain
    verbose=True
)

# ---------------- RUN ----------------
query = "Calculate (50 / 5) + (8 * 6)"
response = agent.run(query)

print("\nFinal Output:")
print(response)
