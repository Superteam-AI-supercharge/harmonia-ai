from langchain.prompts import ChatPromptTemplate
from langchain.chains import TransformChain, SequentialChain
from fuzzywuzzy import process
import json

class TweetAgent:
    def __init__(self, groq_client, handles_path='superteam_data/superteam_vietnam/superteam_vietnam_followed_list.json'):
        self.groq_client = groq_client
        self.handles = self._load_followed_accounts(handles_path)
        self.chain = self._build_chain()

    def _load_followed_accounts(self, path):
        with open(path) as f:
            return {acc['screen_name'].lower(): acc for acc in json.load(f)}

    def _build_chain(self):
        return SequentialChain(
            chains=[
                self._create_draft_chain(),
                self._refine_chain(),
                self._thread_chain()
            ],
            input_variables=["theme"],
            output_variables=["final_threads"]
        )

    def _create_draft_chain(self):
        def generate_drafts(inputs):
            prompt = f"""Generate 5 Twitter thread concepts about {inputs['theme']} for Superteam Vietnam:
            - Include 3-5 tweets per thread
            - Use relevant hashtags (#Web3, #Solana)
            - Mention team members from: {list(self.handles.keys())}
            - Add engaging emojis
            - Each tweet under 280 characters
            
            Format each thread concept as:
            [Thread Title]
            1. [Tweet 1]
            2. [Tweet 2]
            ..."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192"
            )
            return {"drafts": response.choices[0].message.content}
        
        return TransformChain(
            input_variables=["theme"],
            output_variables=["drafts"],
            transform=generate_drafts
        )

    def _refine_chain(self):
        def refine_drafts(inputs):
            corrected = []
            for line in inputs["drafts"].split('\n'):
                for word in line.split():
                    if word.startswith('@'):
                        handle = word[1:].lower()
                        if handle not in self.handles:
                            match, score = process.extractOne(handle, self.handles.keys())
                            if score > 80:
                                line = line.replace(word, f"@{match}")
                corrected.append(line)
            return {"refined": "\n".join(corrected)}
        
        return TransformChain(
            input_variables=["drafts"],
            output_variables=["refined"],
            transform=refine_drafts
        )

    def _thread_chain(self):
        def finalize_threads(inputs):
            prompt = f"""Select and improve the top 3 thread concepts:
            {inputs['refined']}
            
            Requirements:
            1. Add strong opening/closing tweets
            2. Ensure logical flow between tweets
            3. Include 2-3 relevant hashtags per thread
            4. Verify all @mentions are valid
            
            Format each final thread as JSON:
            {{
              "title": "Thread Title",
              "tweets": ["Tweet 1", "Tweet 2"],
              "hashtags": ["#Web3", "#Solana"],
              "mentions": ["@handle1", "@handle2"]
            }}"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                response_format={"type": "json_object"}
            )
            return {"final_threads": json.loads(response.choices[0].message.content)}
        
        return TransformChain(
            input_variables=["refined"],
            output_variables=["final_threads"],
            transform=finalize_threads
        )

    def process_request(self, theme):
        return self.chain({"theme": theme})