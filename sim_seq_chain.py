from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

load_dotenv()

# ✅ FREE TIER MODEL
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.7
)

# Chain 1
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Give a clear definition of {topic}."
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2
prompt2 = PromptTemplate(
    input_variables=["text"],
    template="Explain this in very simple words: {text}"
)

chain2 = LLMChain(llm=llm, prompt=prompt2)

# Sequential chain
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = overall_chain.run("Blockchain")

print("\nFinal Output:\n", result)
