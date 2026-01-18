from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

# Gemini FREE tier model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.7
)

# -------- Chain 1 : Create outline ----------
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Create a short outline for a blog about {topic}."
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="outline")

# -------- Chain 2 : Write blog ----------
prompt2 = PromptTemplate(
    input_variables=["outline"],
    template="Write a short blog using this outline:\n{outline}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="blog")

# -------- Chain 3 : Simplify ----------
prompt3 = PromptTemplate(
    input_variables=["blog"],
    template="Explain this blog in very simple words:\n{blog}"
)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="simple")

# -------- Sequential Chain ----------
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["outline", "blog", "simple"],
    verbose=True
)

# Run
result = overall_chain.invoke({"topic": "Artificial Intelligence"})

print("\n📌 OUTLINE:\n", result["outline"])
print("\n📌 BLOG:\n", result["blog"])
print("\n📌 SIMPLE EXPLANATION:\n", result["simple"])
