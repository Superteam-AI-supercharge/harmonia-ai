import os

# Determine which LLM provider to use.
# Set USE_LOCAL=true in your environment to use the local model (via Ollama)
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() in ("true", "1", "yes")

if USE_LOCAL:
    # Use the Ollama Python package for local inference.
    import ollama

    def get_llm_response(messages, model="llama3:8b"):
        """
        Invoke the local Llama 3-8b model using Ollama.
        Expects messages as a list of dicts with keys "role" and "content".
        """
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=False  # change to True for streaming responses if needed
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "I don't know"
else:
    # Use Groq's client if not using the local model.
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def get_llm_response(messages, model="llama3-70b-8192"):
        """
        Invoke the LLM using Groq.
        Expects messages as a list of dicts with keys "role" and "content".
        """
        try:
            response = groq_client.chat.completions.create(
                messages=messages,
                model=model,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq: {e}")
            return "I don't know"
