# utils.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import requests
url = 'https://6df3-35-197-137-37.ngrok-free.app/testing'

def ai_review(product_title ,product_description):
    myInput = {
        "product_title": product_title,
        "product_description": product_description,
        "target attributes": "RAM"
    }
    # Define the JSON payload
    data = {
        "Task": "Attribute_Value_Extraction",
        "Input": myInput
    }

    response = requests.get(url, json=data)
    print(response.status_code)
    print(response.json())  # Assuming the response is in JSON format
    return response.json()["response"]


# Function to handle LangChain chat with product description
def langchain_chat(product_title ,product_description, question):
    try:

        instruction = "Generate an answer to the question by utilizing the information contained in the document."
        # Define the LangChain template and model
        my_template = f"""
        Answer the question if it is present in the product title, bullet points or description. \
        If question is nonsense, trickery, or has no clear answer, I will respond with "Unknown".
        Start the answer with `A:` and output the answer without any explanation.

        Product Title: {product_title}
        About this item
        {product_description}

        Q: {question}
        """


        model = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_template(my_template)
        llm_chain =  prompt | model | StrOutputParser()

        llm_response = llm_chain.invoke({"product_description": product_description, "question": question, "product_title": product_title})

        return llm_response
    
    except Exception as e:
        print(f"Error generating response from LangChain: {e}")
        return None
    
