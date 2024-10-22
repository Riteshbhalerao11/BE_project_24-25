from fastapi import FastAPI, WebSocket, Body
import json
import uvicorn
from mongo import get_product_by_id
from utils import langchain_chat, ai_review
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from langchain import ChatOpenAI
# from mistral import MistralClient  # hypothetical client; adapt as per your API


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/tasks")
def perform_task(request: dict = Body(...)):
    # product_info = json.loads(request)

    product_id = request.get("id")
    print(product_id)
    product = get_product_by_id(product_id)
    product_description = product.get('description')
    product_title = product.get('title')

    # llm_response = ai_review(product_title, product_description)
    llm_response = ai_review(product_title, product_description)
    return {"message": "11909", "llm_response": llm_response}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        product_data = await websocket.receive_text()
        product_info = json.loads(product_data)

        product_id = product_info.get("id")
        print(product_id)

            # Fetch product description from MongoDB
        product = get_product_by_id(product_id)
        if not product:
            response = {
                    "error": "Product not found",
                    "response": None
            }
            await websocket.send_text(json.dumps(response))
            return

        product_description = product.get('description')
        product_title = product.get('title')
        print(product_title)

        while True:
            data = await websocket.receive_text()
            question_data = json.loads(data)
            question = question_data.get("message")
            print(question)

                # Use LangChain to generate a response using the product description
            llm_response = langchain_chat(product_title, product_description, question)
            print(llm_response)
                # Prepare response
            response = {
                    "original_question": question,
                    "product_id": product_id,
                    "llm_response": llm_response
                }

                # Send the response back to the client
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected")
    
    finally:
        await websocket.close()

def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
