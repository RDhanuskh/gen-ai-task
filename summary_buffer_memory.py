from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.fake import FakeListLLM

# ✅ Fake LLM responses (no API calls)
fake_llm = FakeListLLM(responses=[
    "Hello Dhanush!",
    "Nice, Python is great.",
    "AI is a powerful field.",
    "Chatbots are very useful.",
    "LangChain helps build chatbots.",
    "Yes, your name is Dhanush.",
    "You like Python, AI, chatbots, and LangChain."
])

# ✅ Window memory (recent messages)
recent_memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True
)

conversation = ConversationChain(
    llm=fake_llm,
    memory=recent_memory,
    verbose=False
)

# Manual local summary
conversation_summary = ""

def update_summary(summary, text):
    return summary + " " + text

messages = [
    "Hi, I am Dhanush",
    "I like Python",
    "I am learning AI",
    "I want to build chatbots",
    "LangChain is interesting",
    "Do you remember my name?",
    "What do I like?"
]

for msg in messages:
    print(f"\n💬 User: {msg}")
    response = conversation.invoke(msg)["response"]
    print("🤖 AI:", response)

    conversation_summary = update_summary(conversation_summary, msg)

print("\n📝 Conversation Summary (Old Messages):")
print(conversation_summary)
