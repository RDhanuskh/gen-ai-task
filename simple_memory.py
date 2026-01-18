from langchain.chains import ConversationChain
from langchain_core.memory import BaseMemory
from langchain_community.llms import FakeListLLM
from typing import Dict, Any

# -------------------------------------------------
# Step 1: Custom Simple Memory (ConversationChain-safe)
# -------------------------------------------------
class SimpleMemory(BaseMemory):
    store: Dict[str, Any] = {}

    @property
    def memory_variables(self):
        return ["history"]

    def load_memory_variables(self, inputs):
        history_text = ""
        for k, v in self.store.items():
            history_text += f"{k}: {v}\n"
        return {"history": history_text}

    def save_context(self, inputs, outputs):
        text = inputs.get("input", "")
        text_lower = text.lower()

        # ✅ Extract name only from declarative sentence
        if text_lower.startswith("hi, my name is") or text_lower.startswith("my name is"):
            self.store["name"] = text.split("name is")[-1].strip()

        # ✅ Extract project ONLY when user states it
        if text_lower.startswith("i am working on"):
            self.store["project"] = text.split("working on")[-1].strip()

    def clear(self):
        self.store.clear()

# -------------------------------------------------
# Step 2: Fake LLM (offline, no API)
# -------------------------------------------------
llm = FakeListLLM(responses=[
    "Nice to meet you!",
    "Got it, I have stored that.",
    "Yes, I remember.",
    "You are working on NeoRetinoNet."
])

# -------------------------------------------------
# Step 3: Conversation Chain
# -------------------------------------------------
memory = SimpleMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# -------------------------------------------------
# Step 4: Chat simulation
# -------------------------------------------------
messages = [
    "Hi, my name is Dhanush",
    "I am working on NeoRetinoNet",
    "Do you remember my project?",
    "What project am I working on?"
]

for msg in messages:
    print(f"\n💬 User: {msg}")
    response = conversation.invoke(msg)
    print("🤖 AI:", response["response"])

# -------------------------------------------------
# Step 5: View stored memory
# -------------------------------------------------
print("\n🧠 Stored Simple Memory:")
print(memory.store)
