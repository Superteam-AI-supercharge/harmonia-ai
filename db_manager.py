# db_manager.py
import sqlite3
import json

DB_PATH = "superteam_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            directory TEXT,
            file_name TEXT,
            content TEXT,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_document(doc_id: str, directory: str, file_name: str, content: str, metadata: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    metadata_json = json.dumps(metadata)
    cursor.execute('''
        INSERT INTO documents (id, directory, file_name, content, metadata)
        VALUES (?, ?, ?, ?, ?)
    ''', (doc_id, directory, file_name, content, metadata_json))
    conn.commit()
    conn.close()

def load_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, directory, file_name, content, metadata FROM documents')
    rows = cursor.fetchall()
    conn.close()
    docs = []
    for row in rows:
        doc_id, directory, file_name, content, metadata_json = row
        metadata = json.loads(metadata_json)
        docs.append({
            "id": doc_id,
            "directory": directory,
            "file_name": file_name,
            "content": content,
            "metadata": metadata
        })
    return docs

def delete_documents_by_file(file_name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents WHERE file_name = ?', (file_name,))
    conn.commit()
    conn.close()

