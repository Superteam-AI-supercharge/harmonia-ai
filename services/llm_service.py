from groq import Groq
from llama_cpp import Llama
from typing import Union

class LLMService:
    def __init__(self, config: dict):
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.local_llm = Llama(
            model_path="./models/llama-3-8b.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=4
        ) if config.USE_LOCAL_LLM else None

    def generate(self, prompt: str, use_local: bool = False) -> str:
        if use_local and self.local_llm:
            output = self.local_llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return output['choices'][0]['message']['content']
        else:
            completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.7
            )
            return completion.choices[0].message.content