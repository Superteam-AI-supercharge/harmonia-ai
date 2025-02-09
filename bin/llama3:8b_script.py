# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os, uuid, json
import shutil
import ollama
from model_provider import model_provider
from groq import Groq  # Our LLM client (Groq-hosted Llama)
from document_processor import DocumentProcessor  # Updated processor that handles directories & multiple file types
from db_manager import init_db 
from llm_provider import get_llm_response 
from db_manager import load_all_documents

# LangChain prompt and memory imports
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

load_dotenv()
app = FastAPI()

# Initialize the SQLite database (create table if not exists)
init_db()

superteam_data_path = os.environ.get("SUPERTEAM_DATA_DIRECTORY_PATH")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Initialize Document Processor (builds vector store from directory)
doc_processor = DocumentProcessor()
doc_processor.load_documents_from_db()


# ---------------------------
# Define Prompt Templates and Memory
# ---------------------------
# Prompt template for /learn endpoint (general queries)
# Update the prompt construction in both endpoints

# ---------------------------
# Conversation Memory
# ---------------------------
conversation_memory = {}
# ---------------------------
# Prompt Building Functions
# ---------------------------
def build_learn_messages(query: str, context: str, conversation: list):
    """Build messages for /learn endpoint"""
    system_content = (
        "You are the Superteam Knowledge Bot. Your job is to answer questions about Superteam using only the provided context. "
        "Follow these rules strictly:\n"
        "1. Always include relevant links from source_url metadata when available\n"
        "2. If information is missing from context, say 'I don't know'\n"
        "3. Structure answers clearly with bullet points when listing items\n"
        "4. Always mention the source title and link if available\n"
        f"Context:\n{context}"
    )
    
    messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history
    for msg in conversation:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    messages.append({"role": "user", "content": query})
    return messages

def build_find_messages(query: str, context: str):
    """Build messages for /find endpoint"""
    system_content = (
        "You are the Superteam Member Match Expert. Analyze this member data from the 'superteam_vietnam_members' directory:\n"
        f"{context}\n\n"
        "Rules:\n"
        "1. Match skills/experience exactly from member profiles\n"
        "2. List all matching members with name, skills, and contact info if available. Do not make up contact info.\n"
        "3. Include source links from metadata if available\n"
        "4. If no matches, say 'No matching members found'"
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]

# ---------------------------
# Endpoint 1: Find Superteam Members
# ---------------------------
class FindRequest(BaseModel):
    query: str

@app.post("/find")
async def find_members(request: FindRequest):
    query = request.query
    
    # Search ONLY in superteam_vietnam_members directory
    results = doc_processor.vector_store.similarity_search(
        query, 
        k=10,
        filter={"directory": "superteam_vietnam_members"}
    )
    
    # Format context with member details
    context = []
    for doc in results:
        metadata = doc.metadata
        context.append(
            f"Member: {metadata.get('name', 'Unknown')}\n"
            f"Skills: {', '.join(metadata.get('skills', []))}\n"
            f"Contact: {metadata.get('source_url', 'No contact info available')}\n"
        )
    context_str = "\n".join(context)
    
    # Build structured prompt with full instructions
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Superteam Member Match Expert. Analyze this member data:\n\n"
                "### RULES TO FOLLOW:\n"
                "1. Match skills/experience EXACTLY from member profiles\n"
                "2. List all matching members with name, skills, and contact info (ONLY if available)\n"
                "3. ALWAYS include source links from metadata\n"
                "4. If no matches, clearly state 'No matching members found'\n"
                "5. Never invent information or links not present in the context\n\n"
                "### MEMBER DATA:\n"
                f"{context_str}\n\n"
                "### USER QUERY:\n"
                f"{query}"
            )
        }
    ]
    
    try:
        # Call Ollama locally
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            options = [
                
            ]
        )
        answer = response['message']['content'].strip()
        
        # Fallback handling
        if not answer or "no matching members" in answer.lower():
            answer = "No matching members found based on current data"
            
    except Exception as e:
        print(f"Error generating answer in /find: {e}")
        answer = "Error processing member search"
        
    return {
        "answer": answer,
        "context": context_str
    }

# ---------------------------
# Endpoint 2: Learn (General Query)
# ---------------------------
class LearnRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/learn")
async def learn(request: LearnRequest):
    query = request.query
    session = request.session_id

    # Search the vector store for context
    results = doc_processor.vector_store.similarity_search(query, k=5)
    
    # Build context with sources
    context = []
    sources = set()
    for doc in results:
        source_link = doc.metadata.get('source_url', '')
        context.append(f"Content: {doc.page_content}")
        if source_link:
            sources.add(source_link)
    context_str = "\n\n".join(context)

    # Manage conversation history
    if session not in conversation_memory:
        conversation_memory[session] = []
    conversation = conversation_memory[session]

    # Build base system message with persistent instructions
    system_message = {
        "role": "system",
        "content": (
            "You are the Superteam Knowledge Bot. Follow these RULES STRICTLY:\n\n"
            "1. ANSWER USING ONLY THE PROVIDED CONTEXT\n"
            "2. ALWAYS include relevant links from source_url metadata\n"
            "3. STRUCTURE ANSWERS WITH:\n"
            "   - Clear headings\n"
            "   - Bullet points for lists\n"
            "   - Bold important terms (using **bold** syntax)\n"
            "4. If information is missing or not enough context, say: 'I don't know'\n"
            "5. ALWAYS mention the source title and link when available\n\n"
            "### CURRENT CONTEXT:\n"
            f"{context_str}"
        )
    }

    # Rebuild message history with proper roles
    messages = [system_message]
    for msg in conversation:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Add current query
    messages.append({"role": "user", "content": query})

    try:
        # Get Ollama response
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                'temperature': 0.6,
                'num_predict': 1024,
                'stop': ["\n\n"]  # Prevent overly long responses
            }
        )
        answer = response['message']['content'].strip()
        
        # Append sources if available
        if sources:
            answer += "\n\n**Relevant Links:**\n" + "\n".join(f"- {src}" for src in sources)
        
        # Fallback for empty answers
        if not answer or "i don't know" in answer.lower():
            answer = "I don't have enough information to answer that question"

    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "Error processing request. Please try again."

    # Update conversation history
    conversation.extend([HumanMessage(content=query), AIMessage(content=answer)])
    
    return {
        "answer": answer,
        "context": context_str
    }

# @app.post("/find")
# async def find_members(request: FindRequest):
#     query = request.query
#     print(query)
    
#     # Search ONLY in superteam_vietnam_members directory
#     results = doc_processor.vector_store.similarity_search(
#         query, 
#         k=10,
#         filter={"directory": "superteam_vietnam_members"}
#     )
    
#     # Format context with member details
#     context = []
#     for doc in results:
#         metadata = doc.metadata
#         context.append(
#             f"Member: {metadata.get('name', 'Unknown')}\n"
#             f"Skills: {', '.join(metadata.get('skills', []))}\n"
#             f"Contact: {metadata.get('source_url', 'No contact info available')}\n"
#         )
#     context_str = "\n".join(context)
    
#     # Format the prompt
#     prompt = find_prompt.format(input=query, context=context_str)
    
#     try:
#         chat_completion = groq_client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama3-70b-8192",
#             stream=False,
#         )
#         answer = chat_completion.choices[0].message.content.strip()
        
#         # Fallback if no good response
#         if not answer or "no matching members" in answer.lower():
#             answer = "No matching members found based on current data"
            
#     except Exception as e:
#         print(f"Error generating answer in /find: {e}")
#         answer = "Error processing member search"
#     print(answer)
        
#     return {
#         "answer": answer,
#         "context": context_str
#     }

# # ---------------------------
# # Endpoint 2: Learn (General Query with Conversation Memory)
# # ---------------------------
# class LearnRequest(BaseModel):
#     query: str
#     session_id: str = "default"

# @app.post("/learn")
# async def learn(request: LearnRequest):
#     """Endpoint for general queries with conversation memory"""
#     query = request.query
#     session = request.session_id

#     # Search the vector store for context
#     results = doc_processor.vector_store.similarity_search(query, k=5)
    
#     # Build context with sources
#     context = []
#     sources = set()
#     for doc in results:
#         source_link = doc.metadata.get('source_url', '')
#         context.append(f"Content: {doc.page_content}")
#         if source_link:
#             sources.add(source_link)
#     context_str = "\n\n".join(context)

#     # Manage conversation history
#     if session not in conversation_memory:
#         conversation_memory[session] = []
#     conversation = conversation_memory[session]

#     # Format messages for Groq API
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a Superteam Vietnam expert. Answer using only the context below. "
#                 "Always include relevant links from source_url when available.\n\n"
#                 f"Context:\n{context_str}"
#             )
#         }
#     ]

#     # Add conversation history
#     for msg in conversation:
#         if isinstance(msg, HumanMessage):
#             messages.append({"role": "user", "content": msg.content})
#         elif isinstance(msg, AIMessage):
#             messages.append({"role": "assistant", "content": msg.content})

#     # Add current query
#     messages.append({"role": "user", "content": query})

#     try:
#         # Get Groq response
#         chat_completion = groq_client.chat.completions.create(
#             messages=messages,
#             model="llama3-70b-8192",
#             temperature=0.3,
#             max_tokens=1024
#         )
        
#         answer = chat_completion.choices[0].message.content.strip()
        
#         # Append sources if available
#         if sources:
#             answer += "\n\nRelevant links:\n" + "\n".join(f"- {src}" for src in sources)
        
#         if not answer or "i don't know" in answer.lower():
#             answer = "I don't have enough information to answer that"

#     except Exception as e:
#         print(f"Error generating answer: {e}")
#         answer = "Error processing request"

#     # Update conversation history
#     conversation.extend([HumanMessage(content=query), AIMessage(content=answer)])
    
#     return {
#         "answer": answer,
#         "context": context_str
#     }

# ---------------------------
# Endpoint 3: Upload Documents (Admin Only)
# ---------------------------
@app.post("/upload")
async def upload_document(admin_token: str, file: UploadFile = File(...)):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Create directories for temporary and permanent storage if they don't exist.
    os.makedirs("temp", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Save the uploaded file temporarily.
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Move the file to a permanent location.
    permanent_path = os.path.join("uploads", file.filename)
    shutil.copy(temp_path, permanent_path)  # Copy to permanent folder.
    
    # Process the file using document_processor. This extracts the text, creates Document objects,
    # inserts them into the SQLite database, and updates the vector store.
    doc_processor.process_file(temp_path, base_dir="uploads")
    doc_processor.build_index()  # Update the vector store.
    
    # Remove the temporary file.
    os.remove(temp_path)
    
    return {"message": f"File '{file.filename}' uploaded, processed, and saved permanently at '{permanent_path}'."}

# ---------------------------
# Endpoint 4: Delete Documents by File name (Admin Only)
# ---------------------------

class DeleteByFileRequest(BaseModel):
    admin_token: str
    file_name: str

@app.post("/delete_by_file")
async def delete_by_file(request: DeleteByFileRequest):
    if request.admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    from db_manager import delete_documents_by_file
    delete_documents_by_file(request.file_name)
    
    # Refresh vector store from DB:
    doc_processor.documents = []
    doc_processor.load_documents_from_db()
    
    return {"message": f"All document chunks for file '{request.file_name}' have been deleted."}


# ---------------------------
# Endpoint 5: List all Documents (Admin Only)
# ---------------------------

@app.get("/list")
async def list_documents(admin_token: str):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Load all documents from the SQLite database.
    docs = load_all_documents()
    
    # Group documents by file_name.
    grouped = {}
    for doc in docs:
        key = doc["file_name"]
        # If not already grouped, create a new group.
        if key not in grouped:
            grouped[key] = {
                "file_name": doc["file_name"],
                "directory": doc["directory"]
            }
    
    # Prepare the summary as a list of unique file entries.
    summary = list(grouped.values())
    return {"documents": summary}

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# SELECT * FROM documents;
