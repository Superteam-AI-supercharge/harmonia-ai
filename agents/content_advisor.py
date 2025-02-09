from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class ContentAdvisor:
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.memory = ConversationBufferMemory()
        self.templates = {
            'improve': PromptTemplate(
                input_variables=["content", "platform"],
                template="""Improve this {platform} content:
                {content}
                
                Provide 3 versions that:
                1. Maintain core message
                2. Increase engagement
                3. Platform best practices
                4. Include relevant hashtags/formatting
                
                Return as JSON: {{"versions": ["ver1", "ver2", "ver3"]}}"""
            ),
            'generate_thread': PromptTemplate(
                input_variables=["topic", "points", "platform"],
                template="""Create {platform} thread about {topic}
                Key points: {points}
                
                Requirements:
                1. Strong hook
                2. Break down complex ideas
                3. {platform}-appropriate formatting
                4. Call to action
                
                Return as JSON: {{"thread": ["part1", "part2"]}}"""
            )
        }

    async def improve_content(self, content: str, platform: str = "twitter"):
        prompt = self.templates['improve'].format(
            content=content,
            platform=platform
        )
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def generate_thread(self, topic: str, points: list, platform: str = "twitter"):
        prompt = self.templates['generate_thread'].format(
            topic=topic,
            points="\n- ".join(points),
            platform=platform
        )
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def refine_content(self, session_id: str, feedback: str):
        history = self.memory.load_memory_variables({"session_id": session_id})
        prompt = f"""Collaborative refinement session:
        History: {history}
        New feedback: {feedback}
        
        Generate 2 improved versions addressing the feedback"""
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        return response.choices[0].message.content.split('\n')