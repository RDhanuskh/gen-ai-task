from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ✅ Buffer Window Memory (stores only last 5 messages)
memory = ConversationBufferWindowMemory(
    k=5,                 # Number of messages to remember
    return_messages=True
)

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Chat simulation
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
    response = conversation.invoke(msg)
    print("🤖 AI:", response["response"])
