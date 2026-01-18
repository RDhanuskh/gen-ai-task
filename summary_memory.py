from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini model (used ONLY for replies)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ✅ Buffer memory (NO extra Gemini calls)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Chat
print("💬 User: Hi, I am Dhanush")
print("🤖 AI:", conversation.invoke("Hi, I am Dhanush")["response"])

print("\n💬 User: I like Python and AI")
print("🤖 AI:", conversation.invoke("I like Python and AI")["response"])

print("\n💬 User: Do you remember my name?")
print("🤖 AI:", conversation.invoke("Do you remember my name?")["response"])

print("\n💬 User: What do I like?")
print("🤖 AI:", conversation.invoke("What do I like?")["response"])
