import re

def sentence_counter(text: str) -> str:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    return f"Sentence count: {len(sentences)}"


text = "Hello world. How are you?"
print(sentence_counter(text))  
