from pydantic import BaseModel
from fastapi import UploadFile
from dotenv import load_dotenv
class Auth(BaseModel):
    username:str
    password:str
class Rag(BaseModel):
    jwt:str
    query:str
    content:UploadFile
    
 