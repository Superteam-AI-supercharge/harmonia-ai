# document_processor.py (excerpt)
import os, re, json, csv
from typing import List, Dict, Any
import faiss
import numpy as np
import uuid
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from PyPDF2 import PdfReader
import docx
from db_manager import insert_document, load_all_documents  # Import our DB functions

MODEL_NAME = "all-MiniLM-L6-v2"

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = MODEL_NAME, chunk_size: int = 100, chunk_overlap: int = 20):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[Document] = []
        embedding_dim = len(self.embedding_model.embed_query("test"))
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = LCFAISS(
            embedding_function=self.embedding_model,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def preprocess_text(self, text: str) -> str:
        text = text.strip()
        return re.sub(r'\s+', ' ', text)

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # Extraction methods for PDF, DOCX, CSV, etc. (same as before)
    def extract_text_from_pdf(self, filepath: str) -> str:
        text = ""
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
        return text

    def extract_text_from_docx(self, filepath: str) -> str:
        try:
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            return ""

    def extract_text_from_csv(self, filepath: str) -> str:
        text = ""
        try:
            with open(filepath, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    text += " | ".join(row) + "\n"
        except Exception as e:
            print(f"Error reading CSV {filepath}: {e}")
        return text

    def process_file(self, filepath: str, base_dir: str) -> None:
        ext = os.path.splitext(filepath)[1].lower()
        file_name = os.path.basename(filepath)
        metadata = {
            "title": file_name,
            "directory": base_dir,
            "source": base_dir,
            "tags": []
        }
        content = ""
        if ext == ".json":
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON {filepath}: {e}")
                return
            if isinstance(data, list):
                for entry in data:
                    entry_metadata = metadata.copy()
                    entry_metadata.update({
                        "title": entry.get("title", file_name),
                        "region": entry.get("region", "Unknown"),
                        "source_url": entry.get("source_url", "Unknown"),
                        "tags": entry.get("tags", [])
                    })
                    description = self.preprocess_text(entry.get("description", ""))
                    self.add_document(description, entry_metadata, file_name)
            elif isinstance(data, dict):
                entry_metadata = metadata.copy()
                entry_metadata.update({
                    "title": data.get("title", file_name),
                    "region": data.get("region", "Unknown"),
                    "source_url": data.get("source_url", "Unknown"),
                    "tags": data.get("tags", [])
                })
                description = self.preprocess_text(data.get("description", ""))
                self.add_document(description, entry_metadata, file_name)
            return
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        elif ext == ".pdf":
            content = self.extract_text_from_pdf(filepath)
        elif ext in [".doc", ".docx"]:
            content = self.extract_text_from_docx(filepath)
        elif ext == ".csv":
            content = self.extract_text_from_csv(filepath)
        else:
            print(f"Unsupported file type: {filepath}")
            return

        content = self.preprocess_text(content)
        self.add_document(content, metadata, file_name)

    def add_document(self, content: str, metadata: Dict[str, Any], file_name: str) -> None:
        chunks = self.chunk_text(content)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata=metadata)
            self.documents.append(doc)
            # Generate a unique ID for the document chunk.
            doc_id = str(uuid.uuid4())
            # Insert into the SQLite DB.
            from db_manager import insert_document  # Import here to avoid circular imports
            insert_document(doc_id, metadata.get("directory", "unknown"), file_name, chunk, metadata)
        # Note: Do not rebuild index on every addition.

    def build_index(self) -> None:
        if not self.documents:
            return
        self.vector_store.add_documents(documents=self.documents)
    
    def add_documents_from_directory(self, root_dir: str) -> None:
        file_count = 0
        for subdir, _, files in os.walk(root_dir):
            folder_name = os.path.basename(subdir)
            for file in files:
                filepath = os.path.join(subdir, file)
                self.process_file(filepath, base_dir=folder_name)
                file_count += 1
                print(f"Processed file: {filepath}")
        print(f"Total files processed: {file_count}")
        self.build_index()

    def load_documents_from_db(self) -> None:
        from db_manager import load_all_documents
        docs = load_all_documents()
        for doc_data in docs:
            metadata = doc_data["metadata"]
            doc = Document(page_content=doc_data["content"], metadata=metadata)
            self.documents.append(doc)
        self.build_index()

# For testing purposes:
if __name__ == "__main__":
    # Initialize DB first
    from db_manager import init_db
    init_db()
    dp = DocumentProcessor()
    # Alternatively, you could load from DB:
    dp.load_documents_from_db()
    results = dp.search("Tell me about superteam generally")
    print("Search results:")
    for doc in results:
        print(f"Title: {doc.metadata.get('title')}, Content: {doc.page_content[:100]}...")





















# # document_processor.py
# import os
# import re
# import json
# from typing import List, Tuple, Dict, Any
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # For PDF extraction
# from PyPDF2 import PdfReader
# # For DOCX extraction
# import docx

# MODEL_NAME = "all-MiniLM-L6-v2"  # Choose any suitable SentenceTransformer model

# class DocumentProcessor:
#     def __init__(self, embedding_model=MODEL_NAME, chunk_size=100, chunk_overlap=20):
#         # Initialize the embedding model.
#         self.model = SentenceTransformer(embedding_model)
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.index = None  # This will hold our FAISS index
#         # Each document is stored as a tuple: (doc_id, metadata, chunk, embedding)
#         self.documents: List[Tuple[str, Dict[str, Any], str, np.ndarray]] = []

#     def preprocess_text(self, text: str) -> str:
#         """Clean and normalize text."""
#         text = text.strip()
#         text = re.sub(r'\s+', ' ', text)
#         return text

#     def chunk_text(self, text: str) -> List[str]:
#         """Split text into chunks based on a fixed number of words."""
#         words = text.split()
#         chunks = []
#         start = 0
#         while start < len(words):
#             end = min(start + self.chunk_size, len(words))
#             chunk = ' '.join(words[start:end])
#             chunks.append(chunk)
#             # Move start pointer with overlap.
#             start += self.chunk_size - self.chunk_overlap
#         return chunks

#     # --- File Extraction Methods ---
#     def extract_text_from_txt(self, filepath: str) -> str:
#         """Extract text from a plain text file."""
#         with open(filepath, 'r', encoding='utf-8') as f:
#             return f.read()

#     def extract_text_from_pdf(self, filepath: str) -> str:
#         """Extract text from a PDF file using PyPDF2."""
#         text = ""
#         try:
#             reader = PdfReader(filepath)
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         except Exception as e:
#             print(f"Error reading PDF file {filepath}: {e}")
#         return text

#     def extract_text_from_docx(self, filepath: str) -> str:
#         """Extract text from a DOCX file using python-docx."""
#         try:
#             doc = docx.Document(filepath)
#             full_text = []
#             for para in doc.paragraphs:
#                 full_text.append(para.text)
#             return "\n".join(full_text)
#         except Exception as e:
#             print(f"Error reading DOCX file {filepath}: {e}")
#             return ""

#     def process_file(self, filepath: str, base_dir: str) -> None:
#         """
#         Process a single file (JSON, txt, pdf, or docx) and add its contents to the documents list.
#         base_dir is the directory (or folder name) where the file was found.
#         """
#         ext = os.path.splitext(filepath)[1].lower()
#         file_name = os.path.basename(filepath)
#         # Prepare default metadata using file name and directory.
#         metadata = {
#             "title": file_name,
#             "directory": base_dir,
#             "source_url": "Uploaded file",  # Could be modified if available.
#             "source": base_dir,
#             "tags": []
#         }
#         content = ""
#         if ext == ".json":
#             # For JSON files, assume they contain a list of entries.
#             try:
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 if isinstance(data, list):
#                     for entry in data:
#                         # Merge file-level metadata with entry-specific metadata.
#                         entry_metadata = metadata.copy()
#                         entry_metadata.update({
#                             "title": entry.get("title", file_name),
#                             "region": entry.get("region", "Unknown"),
#                             "source_url": entry.get("source_url", metadata["source_url"]),
#                             "source": entry.get("source", metadata["source"]),
#                             "tags": entry.get("tags", [])
#                         })
#                         description = self.preprocess_text(entry.get("description", ""))                        
#                         self.add_content(file_name, entry_metadata, description)
#                 elif isinstance(data, dict):
#                     # Single JSON object.
#                     entry_metadata = metadata.copy()
#                     entry_metadata.update({
#                         "title": data.get("title", file_name),
#                         "region": data.get("region", "Unknown"),
#                         "source_url": data.get("source_url", metadata["source_url"]),
#                         "source": data.get("source", metadata["source"]),
#                         "tags": data.get("tags", [])
#                     })
#                     description = self.preprocess_text(data.get("description", ""))
#                     self.add_content(file_name, entry_metadata, description)
#             except Exception as e:
#                 print(f"Error processing JSON file {filepath}: {e}")
#             return

#         elif ext == ".txt":
#             content = self.extract_text_from_txt(filepath)
#         elif ext == ".pdf":
#             content = self.extract_text_from_pdf(filepath)
#         elif ext in [".doc", ".docx"]:
#             content = self.extract_text_from_docx(filepath)
#         else:
#             print(f"Unsupported file type: {filepath}")
#             return

#         content = self.preprocess_text(content)
#         self.add_content(file_name, metadata, content)

#     def add_content(self, doc_id: str, metadata: Dict[str, Any], content: str) -> None:
#         """Split content into chunks, generate embeddings, and add to the documents list."""
#         chunks = self.chunk_text(content)
#         for chunk in chunks:
#             embedding = self.model.encode(chunk)
#             self.documents.append((doc_id, metadata, chunk, embedding))
#         self.build_index()  # Optionally rebuild index after adding new content

#     def add_documents_from_directory(self, root_dir: str) -> None:
#         """
#         Traverse the given root directory and process all files.
#         Files in subdirectories will have the subdirectory name as part of their metadata.
#         """
#         for subdir, _, files in os.walk(root_dir):
#             # Use the last part of the subdir path as the directory name.
#             folder_name = os.path.basename(subdir)
#             for file in files:
#                 filepath = os.path.join(subdir, file)
#                 self.process_file(filepath, base_dir=folder_name)

#     def add_json_dataset(self, json_filepath: str) -> None:
#         """
#         If a JSON dataset file is provided (flat file), process it.
#         """
#         try:
#             with open(json_filepath, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             if isinstance(data, list):
#                 for entry in data:
#                     metadata = {
#                         "title": entry.get("title", "Unknown"),
#                         "region": entry.get("region", "Unknown"),
#                         "source_url": entry.get("source_url", "Unknown"),
#                         "source": entry.get("source", "Unknown"),
#                         "tags": entry.get("tags", [])
#                     }
#                     description = self.preprocess_text(entry.get("description", ""))
#                     self.add_content(metadata["title"], metadata, description)
#             elif isinstance(data, dict):
#                 metadata = {
#                     "title": data.get("title", "Unknown"),
#                     "region": data.get("region", "Unknown"),
#                     "source_url": data.get("source_url", "Unknown"),
#                     "source": data.get("source", "Unknown"),
#                     "tags": data.get("tags", [])
#                 }
#                 description = self.preprocess_text(data.get("description", ""))
#                 self.add_content(metadata["title"], metadata, description)
#         except Exception as e:
#             print(f"Error processing JSON dataset {json_filepath}: {e}")

#     def build_index(self) -> None:
#         """Build or rebuild the FAISS index from the document embeddings."""
#         if not self.documents:
#             return
#         embeddings = np.array([doc[3] for doc in self.documents]).astype("float32")
#         self.index = faiss.IndexFlatL2(embeddings.shape[1])
#         self.index.add(embeddings)

#     def search(self, query: str, top_k=5) -> List[Dict[str, Any]]:
#         """
#         Search the index for the top_k chunks relevant to the query.
#         Returns a list of dictionaries with keys: doc_id, metadata, and chunk.
#         """
#         if self.index is None:
#             raise ValueError("Index has not been built yet.")
#         query_embedding = self.model.encode(query).astype("float32")
#         distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), top_k)
#         results = []
#         for idx in indices[0]:
#             if idx < len(self.documents):
#                 doc_id, metadata, chunk, _ = self.documents[idx]
#                 results.append({
#                     "doc_id": doc_id,
#                     "metadata": metadata,
#                     "chunk": chunk
#                 })
#         return results

# # For testing purposes:
# if __name__ == "__main__":
#     dp = DocumentProcessor()
#     # To process an entire directory structure, e.g., 'data/':
#     dp.add_documents_from_directory("superteam_directory/")
#     query = "What is Superteam Nigeria Discord?"
#     results = dp.search(query)
#     print("Search results:")
#     for res in results:
#         print(f"Doc ID: {res['doc_id']}, Title: {res['metadata'].get('title')}, Chunk: {res['chunk'][:100]}...")


