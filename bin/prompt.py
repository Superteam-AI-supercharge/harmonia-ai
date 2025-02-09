# main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from document_processor import DocumentProcessor
import os
from groq import Groq  # Import the Groq client
import json

# LangChain imports
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()
app = FastAPI()

doc_processor = DocumentProcessor()

data_directory = "data"
if os.path.exists(data_directory):
    doc_processor.add_documents_from_directory(data_directory)
else:
    print(f"Data directory not found at {data_directory}")

# Convert each stored document chunk into a LangChain Document.
lc_documents = []
for doc_id, metadata, chunk, embedding in doc_processor.documents:
    # Create a Document using the chunk as page_content and the metadata dictionary.
    lc_documents.append(Document(page_content=chunk, metadata=metadata))

# Create a LangChain FAISS vector store.
embedding_model_name = "all-MiniLM-L6-v2"  # Same model as in your DocumentProcessor.
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
vectorstore = FAISS.from_documents(lc_documents, embeddings)


groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
conversation_memory = {}  # In production, consider a persistent store like Redis.

def generate_answer(query: str, context: str, conversation_history: str) -> str:
    """
    Generate an answer using the Llama 3.3 70B model on Groq Cloud.
    The prompt includes conversation history, retrieved context, and the current query.
    If context is insufficient or the model produces a vague response, a fallback is returned.
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
        if not answer or "i don't know" in answer.lower():
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

    retrieved_docs = vectorstore.similarity_search(query, k=5)

    context = "\n".join([
        (
            f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            f"Region: {doc.metadata.get('region', 'Unknown')}\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Source URL: {doc.metadata.get('source_url', 'Unknown')}\n"
            f"Tags: {', '.join(doc.metadata.get('tags', []))}\n"
            f"Content: {doc.page_content}\n"
        )
        for doc in retrieved_docs
    ])

    conv_history = conversation_memory.get(session, "")
    
    answer = generate_answer(query, context, conv_history)
    
    # Update conversation memory.
    new_entry = f"Q: {query}\nA: {answer}\n"
    conversation_memory[session] = conv_history + "\n" + new_entry if conv_history else new_entry

    return {"answer": answer, "context": context}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading documents (PDF, DOCX, TXT, JSON) to process and add to the index.
    Uploaded files are processed by the DocumentProcessor, then (optionally) re-indexed.
    """
    try:
        # Save the uploaded file temporarily.
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        doc_id = os.path.splitext(file.filename)[0]
        doc_processor.add_document(file_path, doc_id)

        new_docs = []
        for d in doc_processor.documents:
            if d[0] == doc_id:
                new_docs.append(Document(page_content=d[2], metadata=d[1]))
        if new_docs:
            vectorstore.add_documents(new_docs)
        
        os.remove(file_path)
        
        return {"message": f"File '{file.filename}' processed and added to the index."}
    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": f"Failed to process file: {str(e)}"}

# To run the API, use:
# uvicorn main:app --reload
