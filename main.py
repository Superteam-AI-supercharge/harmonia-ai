# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os, uuid, json
import shutil
from groq import Groq  # Our LLM client (Groq-hosted Llama)
from document_processor import DocumentProcessor  # Updated processor that handles directories & multiple file types
from db_manager import init_db 
from db_manager import load_all_documents
from models.schemas import DraftEdit, DraftRequest
from agents.tweet_agent import TweetAgent
from agents.content_advisor import ContentAdvisor

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

init_db()

superteam_data_path = os.environ.get("SUPERTEAM_DATA_DIRECTORY_PATH")
ADMIN_TOKEN_STRINGS = os.environ.get("ADMIN_TOKEN")
ADMIN_TOKENS_ARRAY = ADMIN_TOKEN_STRINGS.split(',')
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

tweet_agent = TweetAgent(groq_client)
content_advisor = ContentAdvisor(groq_client)



# Initialize Document Processor (builds vector store from directory)
doc_processor = DocumentProcessor()
doc_processor.load_documents_from_db()

def admin_checks(admin_token):
    if admin_token not in ADMIN_TOKEN_STRINGS:
        raise HTTPException(status_code=403, detail="Not authorized")

class TweetRequest(BaseModel):
    theme: str
    admin_token: str

# ---------------------------
# Define Prompt Templates and Memory
# ---------------------------
# Prompt template for /learn endpoint (general queries)
learn_system_template = SystemMessagePromptTemplate.from_template(
    template=(
        "You are the Superteam Knowledge Bot. Your job is to answer questions about Superteam using only the provided context. "
        "Follow these rules strictly:\n"
        "1. Always include relevant links from source_url metadata when available\n"
        "2. If information is missing from context, say 'I don't know'\n"
        "3. Structure answers clearly with bullet points when listing items\n"
        "4. Always mention the source title and link if available\n"
        "Context:\n{context}"
    )
)
# Prompt template for /learn endpoint (finding superteam members)
find_system_template = SystemMessagePromptTemplate.from_template(
    template=(
        "You are the Superteam Member Match Expert. Analyze this member data from the 'superteam_vietnam_members' directory:\n"
        "{context}\n\n"
        "Rules:\n"
        "1. Match skills/experience exactly from member profiles\n"
        "2. List all matching members with name, skills, and contact info if only available. Do not make up contact info.\n"
        "3. Include source links from metadata if available\n"
        "4. If no matches, say 'No matching members found'\n"
        "User query: {input}"
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

@app.get("/find")
async def find_members(request: FindRequest):
    query = request.query
    print(query)
    
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
    
    # Format the prompt
    prompt = find_prompt.format(input=query, context=context_str)
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            stream=False,
        )
        answer = chat_completion.choices[0].message.content.strip()
        
        # Fallback if no good response
        if not answer or "no matching members" in answer.lower():
            answer = "No matching members found based on current data"
            
    except Exception as e:
        print(f"Error generating answer in /find: {e}")
        answer = "Error processing member search"
    print(answer)
        
    return {
        "answer": answer,
        "context": context_str
    }

# ---------------------------
# Endpoint 2: Learn (General Query with Conversation Memory)
# ---------------------------
class LearnRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/learn")
async def learn(request: LearnRequest):
    """Endpoint for general queries with conversation memory"""
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

    # Format messages for Groq API
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Superteam Vietnam expert. Answer using only the context below. "
                "Always include relevant links from source_url when available.\n\n"
                f"Context:\n{context_str}"
            )
        }
    ]

    # Add conversation history
    for msg in conversation:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Add current query
    messages.append({"role": "user", "content": query})

    try:
        # Get Groq response
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1024
        )
        
        answer = chat_completion.choices[0].message.content.strip()
        
        # Append sources if available
        if sources:
            answer += "\n\nRelevant links:\n" + "\n".join(f"- {src}" for src in sources)
        
        if not answer or "i don't know" in answer.lower():
            answer = "I don't have enough information to answer that"

    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "Error processing request"

    # Update conversation history
    conversation.extend([HumanMessage(content=query), AIMessage(content=answer)])
    
    return {
        "answer": answer,
        "context": context_str
    }

# ---------------------------
# Endpoint 3: Upload Documents (Admin Only)
# ---------------------------
@app.post("/upload")
async def upload_document(admin_token: str, file: UploadFile = File(...)):
    admin_checks(admin)
    
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
    admin_checks(admin_token)
    
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
    admin_checks(admin_token)
    
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


@app.post("/tweet-pipeline")
async def tweet_pipeline(request: TweetRequest):
    # Verify admin access
    admin_checks(request.admin_token)
    
    # Execute full pipeline
    result = tweet_agent.process_request(request.theme)
    
    
    return {
        "status": "success",
        "threads": result["final_threads"]
    }

class ImproveRequest(BaseModel):
    content: str
    platform: str = "twitter"
    session_id: str = "default"

class ThreadRequest(BaseModel):
    topic: str 
    points: list[str]
    platform: str = "twitter"

@app.post("/improve-content")
async def improve_content(request: ImproveRequest):
    result = await content_advisor.improve_content(
        request.content, 
        request.platform
    )
    return {"versions": result["versions"]}

@app.post("/generate-thread")
async def generate_thread(request: ThreadRequest):
    result = await content_advisor.generate_thread(
        request.topic,
        request.points,
        request.platform
    )
    return {"thread": result["thread"]}

@app.post("/refine-content")
async def refine_content(session_id: str, feedback: str):
    result = await content_advisor.refine_content(session_id, feedback)
    return {"suggestions": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)