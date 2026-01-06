
# Simple AI Agent - No LangChain


from transformers import pipeline


def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Error in calculation"


knowledge_base = {
    "ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "agent": "An AI agent is a system that can perceive, decide, and act to achieve a goal.",
    "rag": "RAG stands for Retrieval-Augmented Generation, combining retrieval and generation.",
    "python": "Python is a high-level programming language widely used in AI."
}

def knowledge_search(query):
    return knowledge_base.get(query.lower(), "No information found.")


print("Loading language model...")
llm = pipeline(
    "text-generation",
    model="gpt2"
)
print("Model loaded successfully!")


def agent(user_input):
    print("\n[Agent Thinking...]")

    
    if "calculate" in user_input.lower():
        expression = user_input.lower().replace("calculate", "").strip()
        result = calculator(expression)
        return f" Calculation Result: {result}"

    elif user_input.lower().startswith("what is"):
        topic = user_input.lower().replace("what is", "").strip()
        result = knowledge_search(topic)
        return f"📘 Knowledge: {result}"

    else:
        # Fallback to LLM
        response = llm(user_input, max_new_tokens=60)
        return response[0]["generated_text"]


if __name__ == "__main__":
    print("\n AI Agent is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Agent stopped. ")
            break

        output = agent(user_input)
        print("Agent:", output)
