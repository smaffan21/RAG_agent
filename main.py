# import Ollama LLM & prompt template
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# import retriever from vector.py file
from vector import retriever

# using llama 3.2
model = OllamaLLM(model="llama3.2")

# prompt template for q&a
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# create a chat prompt template (structured method to create prompts for llm in langchain)
prompt = ChatPromptTemplate.from_template(template)
# chain where the | operator creates pipeline between the prompt and model (it 1st processes the prompt and then feeds it to the llm)
chain = prompt | model

while True: # user input loop
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    # retrieveing reviews based on the question (semantic search, 1 embedding vector/question, cosine similarity, top k reviews)
    reviews = retriever.invoke(question)
    # invoke chain with the reviews & question to get an answer
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)