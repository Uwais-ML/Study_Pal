from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from database import DATABASE
from openai import AsyncOpenAI
import os
import asyncio
load_dotenv()
Db=DATABASE()
import jwt
app = FastAPI()
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        query="SELECT 1 FROM users WHERE username = ? LIMIT 1"
        Present=Db.command(query,(username,))
        if Present is None :
            raise credentials_exception
        return {"username": username, "scopes": payload.get("scopes", [])}
        
    except jwt.PyJWTError:  
        raise credentials_exception

    
    
