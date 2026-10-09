from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from Ingestion import Ingestion
from Query_writer import Qwery_writer

class Retrival:
    def __init__(self,path,api_key="None",url=None):
        self.api_key= os.getenv("GROQ_API_KEY", "None")
        self.url=os.getenv("GROQ_API_BASE","None")
        self.llm=ChatOpenAI(
            model="groq",
            api_key=self.api_key,
            temperature=0.3
        )
        self.ingest=Ingestion(path)
        self.retriever=self.ingest.ingest()
        self.qwery=Qwery_writer()
    
    
    async def retrieve(self,query):
        Structured_qwery=self.qwery.qwery_reponse(query)
        prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the context:\n\n{context}"),
    ("human", "{question}")
])
        rag_chain = (
    {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
     "question": lambda x: x}

    | prompt | self.llm | StrOutputParser()
)
        async for chunk in rag_chain.astream(Structured_qwery):
            yield chunk