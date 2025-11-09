from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from typing import Dict, Any

class MovieRecommenderAgent:
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        self.llm = OpenAI(openai_api_key=openai_api_key)
        pinecone.init(api_key=pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = Pinecone.from_existing_index(
            "movies",
            self.embeddings
        )
    
    def get_context(self, query: str) -> Dict[str, Any]:
        relevant_docs = self.vectorstore.similarity_search(query)
        return {
            "documents": relevant_docs,
            "query": query
        }
    
    def generate_response(self, user_query: str) -> str:
        context = self.get_context(user_query)
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Based on the user query: {query}
            And the following context: {context}
            Provide movie recommendations with explanations.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            query=user_query,
            context=context
        )
        return response

if __name__ == "__main__":
    agent = MovieRecommenderAgent(
        openai_api_key="your-key",
        pinecone_api_key="your-key"
    )
    response = agent.generate_response(
        "I like sci-fi movies with time travel themes"
    )
    print(response)
