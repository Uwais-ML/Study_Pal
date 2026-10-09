
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
class Qwery_writer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    
    def qwery_reponse(self,qwery):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You convert the given request into structured qwery"),
            ("human", "{question}")
        ])
        rag_chain = (
    {
     "question": lambda x: x}

    | prompt | self.llm | StrOutputParser()
)   
        response=rag_chain.invoke(qwery)
        if response:
            return response
        return qwery






