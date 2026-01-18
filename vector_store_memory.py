from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import FakeListLLM
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory

# -------------------------------------------------
# Step 1: Embeddings (offline)
# -------------------------------------------------
embeddings = FakeEmbeddings(size=384)

# -------------------------------------------------
# Step 2: Knowledge documents
# -------------------------------------------------
docs = [
    Document(page_content="NeoRetinoNet is a deep learning project for medical image analysis."),
    Document(page_content="NeoRetinoNet is used for detecting Retinopathy of Prematurity.")
]

vector_store = FAISS.from_documents(docs, embeddings)

# -------------------------------------------------
# Step 3: Retriever (k=1)
# -------------------------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

memory = VectorStoreRetrieverMemory(retriever=retriever)

# -------------------------------------------------
# Step 4: Fake LLM (flow only)
# -------------------------------------------------
llm = FakeListLLM(responses=[
    "Retrieved relevant information from memory."
])

# -------------------------------------------------
# Step 5: Conversation Chain
# -------------------------------------------------
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# -------------------------------------------------
# Step 6: Ask questions
# -------------------------------------------------
questions = [
    "What project am I working on?",
    "What is NeoRetinoNet used for?"
]

for q in questions:
    print(f"\n💬 User: {q}")

    # Run conversation (stores context)
    conversation.invoke(q)

    # ✅ Correct retrieval (NO deprecated method)
    retrieved_docs = retriever.invoke(q)

    print("📄 Retrieved Memory:")
    for d in retrieved_docs:
        print("-", d.page_content)
