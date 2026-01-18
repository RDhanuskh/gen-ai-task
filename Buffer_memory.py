# File: Buffer_memory.py

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda

# 1️⃣ Initialize buffer memory
memory = ConversationBufferMemory()

# 2️⃣ Dummy LLM function that reads memory and remembers name
def dummy_llm(prompt: str, **kwargs) -> str:
    # Get conversation history from memory
    history = memory.load_memory_variables({}).get("history", "")
    
    # If user introduces their name
    if "Hi, I am " in prompt:
        name = prompt.split("Hi, I am ")[1].strip()
        # Add to memory
        memory.save_context({"input": prompt}, {"output": f"Nice to meet you, {name}!"})
        return f"Nice to meet you, {name}!"
    
    # If user asks for their name
    elif "Do you remember my name" in prompt:
        if "Hi, I am " in history:
            name = history.split("Hi, I am ")[1].split("\n")[0].strip()
            memory.save_context({"input": prompt}, {"output": f"Yes, your name is {name}."})
            return f"Yes, your name is {name}."
        memory.save_context({"input": prompt}, {"output": "Sorry, I don't remember your name."})
        return "Sorry, I don't remember your name."
    
    else:
        memory.save_context({"input": prompt}, {"output": f"AI response to: {prompt}"})
        return f"AI response to: {prompt}"

# 3️⃣ Wrap dummy LLM in RunnableLamba
llm = RunnableLambda(dummy_llm)

# 4️⃣ Simple conversation function (no Chain templates)
def chat(user_input: str) -> str:
    return llm.invoke(user_input)

# 5️⃣ Run conversation
print("💬 User: Hi, I am Dhanush")
response1 = chat("Hi, I am Dhanush")
print("🤖 AI:", response1)

print("\n💬 User: Do you remember my name?")
response2 = chat("Do you remember my name?")
print("🤖 AI:", response2)
