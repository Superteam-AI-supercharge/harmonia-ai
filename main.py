# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os, uuid, json
from groq import Groq  # Our LLM client (Groq-hosted Llama)
from document_processor import DocumentProcessor  # Updated processor that handles directories & multiple file types

# LangChain prompt and memory imports
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

load_dotenv()
app = FastAPI()

# Initialize Document Processor (builds vector store from directory)
doc_processor = DocumentProcessor()
doc_processor.add_documents_from_directory("/Users/favourolaboye/Documents/Test/superteam_directory")
# The vector store is accessible via: doc_processor.vector_store

# Initialize the Groq client with the API key from the environment.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Admin token for protected endpoints.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "secret_admin")

# ---------------------------
# Define Prompt Templates and Memory
# ---------------------------
# Prompt template for /learn endpoint (general queries)
learn_system_template = SystemMessagePromptTemplate.from_template(
    template=(
        "You are an expert on Superteam. Use only the provided context to answer the query. "
        "If the context does not contain enough information, reply with 'I don't know'."
    )
)
learn_human_template = HumanMessagePromptTemplate.from_template(template="{input}")
learn_prompt = ChatPromptTemplate.from_messages(
    [learn_system_template, MessagesPlaceholder(variable_name="history"), learn_human_template]
)

# Prompt template for /find endpoint (finding superteam members)
find_system_template = SystemMessagePromptTemplate.from_template(
    template=(
        "You are an expert on Superteam members. Answer strictly based on the documented data. "
        "If you do not have relevant information, reply with 'I don't know'."
    )
)
find_human_template = HumanMessagePromptTemplate.from_template(template="{input}")
find_prompt = ChatPromptTemplate.from_messages([find_system_template, find_human_template])

# ---------------------------
# In-memory conversation memory: dictionary mapping session_id to list of message objects.
conversation_memory = {}

# ---------------------------
# Endpoint 1: Find Superteam Members
# ---------------------------
class FindRequest(BaseModel):
    query: str

@app.post("/find")
async def find_members(request: FindRequest):
    query = request.query
    # Perform similarity search in the vector store.
    results = doc_processor.vector_store.similarity_search(query, k=5)
    context = "\n".join(
        [f"Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in results]
    )
    # Format the prompt using the find prompt template.
    prompt = find_prompt.format(input=f"Query: {query}\nContext:\n{context}")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            stream=False,
        )
        answer = chat_completion.choices[0].message.content.strip()
        if not answer or "i don't know" in answer.lower():
            answer = "I don't know"
    except Exception as e:
        print(f"Error generating answer in /find: {e}")
        answer = "I don't know"
    return {"answer": answer, "context": context}

# ---------------------------
# Endpoint 2: Learn (General Query with Conversation Memory)
# ---------------------------
class LearnRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/learn")
async def learn(request: LearnRequest):
    query = request.query
    print(query)
    session = request.session_id

    # Search the vector store for context.
    results = doc_processor.vector_store.similarity_search(query, k=5)
    context = "\n".join(
        [f"Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in results]
    )
    
    # Ensure conversation memory for this session is a list of message objects.
    if session not in conversation_memory:
        conversation_memory[session] = []
    conversation = conversation_memory[session]
    
    # Append the current user query to the conversation history.
    conversation.append(HumanMessage(content=query))
    
    # Build the prompt using the learn prompt template.
    # We pass the conversation history (a list of base messages) as "history".
    prompt = learn_prompt.format(input=query, history=conversation + [HumanMessage(content=f"Context:\n{context}")])
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            stream=False,
        )
        answer = chat_completion.choices[0].message.content.strip()
        print(answer)
        if not answer or "i don't know" in answer.lower():
            answer = "I don't know"
    except Exception as e:
        print(f"Error generating answer in /learn: {e}")
        answer = "I don't know"
    
    # Append the agent's answer to the conversation history.
    conversation.append(AIMessage(content=answer))
    
    return {"answer": answer, "context": context}

# ---------------------------
# Endpoint 3: Upload Documents (Admin Only)
# ---------------------------
@app.post("/upload")
async def upload_document(admin_token: str, file: UploadFile = File(...)):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Save the uploaded file temporarily.
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Process file using document_processor.
    doc_processor.process_file(temp_path, base_dir="uploads")
    doc_processor.build_index()  # Rebuild the index after adding new documents.
    
    os.remove(temp_path)
    return {"message": f"File '{file.filename}' uploaded and processed."}

# ---------------------------
# Endpoint 4: Delete Documents (Admin Only)
# ---------------------------
class DeleteRequest(BaseModel):
    admin_token: str
    doc_ids: list[str]

@app.post("/delete")
async def delete_document(request: DeleteRequest):
    if request.admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    doc_processor.vector_store.delete(ids=request.doc_ids)
    return {"message": "Documents deleted successfully."}

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
