Superteam AI Communication Assistant
This project is an AI-driven solution for Superteam communities. It combines a FastAPI backend, advanced document processing, LangChain-powered retrieval, and a Groq-based LLM to provide context-aware answers from a diverse dataset. The system supports multiple file formats (JSON, TXT, PDF, DOCX) and organizes data in a directory structure for different Superteam groups (e.g., superteam, solana, superteam_vn, superteam_nig, bounties, grants, hackathons).

Features
Document Processing:

Traverse directory structures and process various file types.
Extract text, images, links, and metadata.
Convert scraped text or uploaded documents into structured JSON and LangChain Documents.
Advanced Retrieval:

Index document chunks using FAISS with SentenceTransformer embeddings.
Perform similarity search to retrieve the most relevant content based on a query.
LLM Integration:

Use a Groq-hosted Llama 3.3 70B (or similar) model to generate answers based on a combination of retrieved context and conversation history.
Conversation Memory:

Maintain in-memory conversation history per session for context-aware responses.
Admin Upload Functionality:

An admin-only endpoint and Telegram bot integration for uploading documents to specific directories.
Telegram Bot Integration:

A Telegram bot that queries the FastAPI backend and returns answers to user questions.
Requirements
Python 3.8+
FastAPI – Web framework.
Uvicorn – ASGI server.
python-telegram-bot – Telegram bot integration.
LangChain – For document abstraction and vector store integration.
faiss-cpu – For vector indexing and similarity search.
sentence-transformers – For generating embeddings.
PyPDF2 – For PDF text extraction.
python-docx – For DOCX file extraction.
python-dotenv – For managing environment variables.
groq – Groq client for LLM API calls. ( For now)

Installation
1. Clone the Repository:
[git clone https://github.com/yourusername/superteam-ai-communication-assistant.git]
cd harmonia-ai

2. Create and Activate a Virtual Environment:
python -m venv newenv
source newenv/bin/activate  # On Windows: newenv\Scripts\activate

3. Install Dependencies:
pip install fastapi uvicorn python-telegram-bot python-dotenv langchain faiss-cpu sentence-transformers PyPDF2 python-docx groq

4. Set Up Environment Variables:
Create a .env file in the project root with the following (adjust as needed):
env
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

5. Directory Structure
├── superteam_directory/                           # Directory containing subfolders for different Superteam groups and files.
│   ├── superteam/
│   ├── solana/
│   ├── superteam_vn/
│   ├── superteam_nig/
│   ├── bounties/
│   ├── grants/
│   └── hackatons/
├── temp/                           # Temporary folder for file uploads.
├── document_processor.py           # Custom document processing module.
├── main.py                         # FastAPI backend with integration for data loading, querying, and file uploads.
├── telegram_bot.py                 # Telegram bot integration.
├── .env                            # Environment variables.
├── README.md                       # This file.

6. Running the Project
FastAPI Backend
To run the FastAPI backend: uvicorn main:app --reload

7. Load this http://127.0.0.1:8000/docs on your browser to test the endpoints below:

/find members Endpoint:
Send POST requests with a JSON body, e.g.:
{
    "query": "What is Superteam Nigeria Discord?",
}


/learn Endpoint:
Send POST requests with a JSON body, e.g.:
{
    "query": "What is Superteam Nigeria Discord?",
    "session_id": "default"
}

/upload Endpoint:
Use this endpoint to upload documents (PDF, DOCX, TXT, JSON). This endpoint is typically restricted to admin users (via additional logic).

/delete Endpoint:
Use this endpoint to delete documents (PDF, DOCX, TXT, JSON). This endpoint is typically restricted to admin users (via additional logic).

Telegram Bot
Run the Telegram bot on another terminal by executing:
python telegram_bot.py ( Make sure the main.py file is also running)


The bot will start polling and respond to user messages by forwarding queries to the FastAPI /query endpoint.
You can chat with the bot on telegram now!