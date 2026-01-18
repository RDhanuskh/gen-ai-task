from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain_community.llms import FakeListLLM
from langchain.prompts import PromptTemplate

# Fake LLM (offline, quota-safe)
fake_llm = FakeListLLM(responses=[
    "Hello Dhanush!",
    "That sounds like an interesting project.",
    "Yes, I remember your project.",
    "You are working on NeoRetinoNet."
])

# Entity Memory
entity_memory = ConversationEntityMemory(llm=fake_llm)

# Custom prompt REQUIRED for entity memory
prompt = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template="""
You are a helpful AI assistant.

Known entities:
{entities}

Conversation history:
{history}

User: {input}
AI:"""
)

# Conversation chain
conversation = ConversationChain(
    llm=fake_llm,
    memory=entity_memory,
    prompt=prompt,
    verbose=False
)

# Chat simulation
messages = [
    "Hi, my name is Dhanush",
    "I am working on NeoRetinoNet",
    "Do you remember my project?",
    "What project am I working on?"
]

for msg in messages:
    print(f"\n User: {msg}")
    result = conversation.invoke({"input": msg})
    print(" AI:", result["response"])

# ✅ Correct way to access stored entities
print("\n Stored Entities (Entity Store):")
print(entity_memory.entity_store.store)
