# main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from document_processor import DocumentProcessor
import os
from groq import Groq  # Import the Groq client
import json

load_dotenv()
app = FastAPI()

# Initialize the document processor.
doc_processor = DocumentProcessor()

# Load the JSON dataset at startup.
json_dataset_path = "dataset.json"  # Path to your cleaned JSON dataset
if os.path.exists(json_dataset_path):
    doc_processor.add_json_dataset(json_dataset_path)
else:
    print(f"JSON dataset not found at {json_dataset_path}")

# Initialize the Groq client with the API key from the environment.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

conversation_memory = {}

def generate_answer(query: str, context: str, conversation_history: str) -> str:
    """
    Generate an answer using the Llama 3.3 70B model on Groq Cloud.
    This function constructs a prompt combining:
      - the conversation history,
      - the retrieved context from the dataset, and
      - the current query.
    If the context is insufficient, it returns a fallback response.
    """
    if not context or len(context) < 20:
        return "I don't have enough information to answer that."

    full_context = f"Conversation History:\n{conversation_history}\nContext:\n{context}"
    prompt = f"Query: {query}\n{full_context}\nAnswer:"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            stream=False,
        )
        answer = chat_completion.choices[0].message.content.strip()
        if not answer or "I don't know" in answer.lower():
            return "I don't have enough information to answer that."
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I don't have enough information to answer that."

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    query = request.query
    session = request.session_id
    results = doc_processor.search(query, top_k=5)
    print(query)


    context = "\n".join([
        (
            f"Title: {item['metadata'].get('title', 'Unknown')}\n"
            f"Region: {item['metadata'].get('region', 'Unknown')}\n"
            f"Source: {item['metadata'].get('source', 'Unknown')}\n"
            f"Source URL: {item['metadata'].get('source_url', 'Unknown')}\n"
            f"Tags: {', '.join(item['metadata'].get('tags', []))}\n"
            f"Content: {item['chunk']}\n"
        )
        for item in results
    ])
    conv_history = conversation_memory.get(session, "")
    
    # Generate an answer using the Groq Llama API, including conversation history.
    answer = generate_answer(query, context, conv_history)
    print(answer)
    
    # Update conversation memory with the new query and answer.
    # For simplicity, we append as plain text. In a production setting, you might store structured data.
    new_entry = f"Q: {query}\nA: {answer}\n"
    conversation_memory[session] = conv_history + "\n" + new_entry if conv_history else new_entry

    return {"answer": answer, "context": context}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading documents (PDF or text files) to process and add to the index.
    """
    try:
        # Save the uploaded file temporarily.
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Add the document to the processor.
        doc_id = os.path.splitext(file.filename)[0]
        doc_processor.add_document(file_path, doc_id)

        # Clean up the temporary file.
        os.remove(file_path)

        return {"message": f"File '{file.filename}' processed and added to the index."}

    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": f"Failed to process file: {str(e)}"}

# To run the API, use:
# uvicorn main:app --reload