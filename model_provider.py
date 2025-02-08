import os
from groq import Groq
import ollama
from dotenv import load_dotenv

load_dotenv()

class ModelProvider:
    def __init__(self):
        self.provider = os.getenv("MODEL_PROVIDER", "local").lower()
        self.groq_client = None
        self.local_model = "llama3:8b"  # Default local model
        
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is required when using Groq provider")
            self.groq_client = Groq(api_key=api_key)
            
    def get_response(self, messages, temperature=0.7, max_tokens=1024):
        if self.provider == "groq":
            return self._groq_chat(messages, temperature, max_tokens)
        return self._local_chat(messages, temperature, max_tokens)
    
    def _groq_chat(self, messages, temperature, max_tokens):
        response = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _local_chat(self, messages, temperature, max_tokens):
        response = ollama.chat(
            model=self.local_model,
            messages=messages,
            options={
                'temperature': temperature,
                'num_predict': max_tokens,
                'top_p': 0.9
            }
        )
        return response['message']['content'].strip()

# Initialize provider
model_provider = ModelProvider()