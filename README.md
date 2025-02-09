# Superteam AI Communication Assistant

## Overview
Superteam AI Communication Assistant is an AI-driven solution designed to facilitate knowledge sharing and document retrieval within Superteam communities. It integrates a FastAPI backend, advanced document processing, LangChain-powered retrieval, and a Groq-based LLM to provide intelligent and context-aware responses.

Additionally, a Telegram bot allows users to interact with the system, upload files, and retrieve information conveniently.

---
## Features

### **Document Processing**
- Traverse directory structures and process various file types (JSON, TXT, PDF, DOCX).
- Extract text, images, links, and metadata.
- Convert scraped text or uploaded documents into structured JSON and LangChain documents.

### **Advanced Retrieval**
- Index document chunks using FAISS with SentenceTransformer embeddings.
- Perform similarity searches to retrieve relevant content based on user queries.

### **LLM Integration**
- Use a Groq-hosted Llama 3.3 70B (or similar) model to generate answers based on retrieved context and conversation history.

### **Conversation Memory**
- Maintain in-memory conversation history per session for more relevant responses.

### **Admin Upload Functionality**
- Admin-only endpoint and Telegram bot integration for uploading documents to specific directories.

### **Telegram Bot Integration**
- A Telegram bot that queries the FastAPI backend and returns intelligent responses to user questions.
- Users can upload and delete files via the bot.
- A command menu enables users to interact with various functionalities.

---
## Requirements

Ensure you have the following installed:
- Python 3.8+
- FastAPI – Web framework.
- Uvicorn – ASGI server.
- python-telegram-bot – Telegram bot integration.
- LangChain – For document abstraction and vector store integration.
- faiss-cpu – For vector indexing and similarity search.
- sentence-transformers – For generating embeddings.
- PyPDF2 – For PDF text extraction.
- python-docx – For DOCX file extraction.
- python-dotenv – For managing environment variables.
- groq – Groq client for LLM API calls.

---
## Installation

### **1. Clone the Repository:**
```sh
git clone https://github.com/yourusername/superteam-ai-communication-assistant.git
cd harmonia-ai
```

### **2. Create and Activate a Virtual Environment:**
```sh
python -m venv newenv
source newenv/bin/activate  # On Windows: newenv\Scripts\activate
```

### **3. Install Dependencies:**
```sh
pip install fastapi uvicorn python-telegram-bot python-dotenv langchain faiss-cpu sentence-transformers PyPDF2 python-docx groq
```

### **4. Set Up Environment Variables:**
Create a `.env` file in the project root and add the following variables:
```env
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
use_local=True
```

---
## Directory Structure
```
├── superteam_data/      # Directory containing subfolders for different Superteam groups.
│   ├── superteam/
│   ├── solana/
│   ├── superteam_vn/
│   └── hackathons/
├── temp/                     # Temporary folder for file uploads.
├── document_processor.py      # Custom document processing module.
├── main.py                    # FastAPI backend with data handling logic.
├── telegram_bot.py            # Telegram bot integration.
├── .env                       # Environment variables.
├── README.md                  # Project documentation.
```

---
## Running the Project

### **1. Start the FastAPI Backend:**
```sh
uvicorn main:app --reload
```

### **2. Access API Documentation:**
Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```

### **3. Available API Endpoints:**

#### **/find_members Endpoint:**
- Send POST requests with a JSON body:
```json
{
    "query": "What is Superteam Nigeria Discord?"
}
```

#### **/learn Endpoint:**
- Send POST requests with a JSON body:
```json
{
    "query": "What is Superteam Nigeria Discord?",
    "session_id": "default"
}
```

#### **/upload Endpoint:**
- Use this endpoint to upload documents (PDF, DOCX, TXT, JSON).
- Typically restricted to admin users.

- #### **/list Endpoint:**
- Use this endpoint to get all documents in the database.
- Typically restricted to admin users.

- #### **/twitter-pipeline:**
- - Send POST requests with a JSON body:
```json
{
   "theme": "I got into BU",
  "admin_token": ""
}
```
- #### **/improve_content:**
- - Send POST requests with a JSON body:
```json
{
  "content": "string",
  "platform": "twitter",
  "session_id": "default"
}
```


#### **/delete Endpoint:**
- Use this endpoint to delete documents (PDF, DOCX, TXT, JSON).
- Typically restricted to admin users.

---

Make sure you have Llama runnning on your local computer.
## Telegram Bot

### **1. Running the Telegram Bot:**
Open a new terminal and run:
```sh
cd telegram
python bot.py
```
(Make sure `main.py` is also running.)

### **2. Interacting with the Bot:**
The bot starts polling and responds to user messages by forwarding queries to the FastAPI `/query` endpoint.

#### **Bot Commands:**
- `/start` - Start the bot and get a welcome message.
- `/upload` - Upload documents (admin-only).
- `/delete` - Delete documents (admin-only).
- `/find_members` - Search for members in Superteam.
- `/help` - Get a list of available commands.

### **3. Chatting with the Bot:**
- Open Telegram and search for your bot.
- Start a conversation and ask questions.
- Upload or delete files using the available commands.

---
