
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore  # Fix: correct import paths
from langchain_classic.retrievers import ParentDocumentRetriever

PARENT_STORE_DIR = "./persisted_parent_chunks"
fs = LocalFileStore(PARENT_STORE_DIR)

class Ingestion:
    def __init__(self, path: str):
        self.path = path  
        self.store = create_kv_docstore(fs)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        
        self.vecstore = Chroma(
             collection_name="My_own_collection",
             embedding_function=self.embeddings,
             persist_directory="./data"
        )

        self.recursive = RecursiveCharacterTextSplitter(
             chunk_size=2000,   
             chunk_overlap=200,
        )

        self.semantic = SemanticChunker(
             embeddings=self.embeddings,
             breakpoint_threshold_type="percentile",
             breakpoint_threshold_amount=90
        )

    def datatype_handler(self):
        if self.path.lower().endswith(".pdf"):
            return PyPDFLoader(self.path)
        elif self.path.lower().endswith((".md", ".txt")): # Consolidated syntax
            return TextLoader(self.path)
        else:
            return None

    def ingest(self):
        loader = self.datatype_handler()
        if not loader:
            raise ValueError(f"Unsupported file format for path: {self.path}")
            
        retriever = ParentDocumentRetriever(
             vectorstore=self.vecstore,
             docstore=self.store,
             child_splitter=self.semantic, # Using semantic chunking for fine-grained vectors
             parent_splitter=self.recursive,
             search_type="mmr",
             search_kwargs={
                 "k": 5,          
                 "fetch_k": 20,    
                 "lambda_mult": 0.5
             }
        )
        
        print(f"Loading and processing: {self.path}")
        documents = loader.load()
        retriever.add_documents(documents)
        return retriever
