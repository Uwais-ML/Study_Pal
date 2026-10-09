from fastapi import FastAPI
from slowapi import Limiter ,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Depends
from typing import Annotated
from config import Rag,Auth
from fastapi.responses import StreamingResponse
from dependencies import get_current_user
from Rag.Retrival import Retrival
from security import login,signin
limiter=Limiter(key_func=get_remote_address)
app=FastAPI()
app.state.limiter=limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)
@app.get("/health")
@limiter.limit("5/minute")
def health_check():
    return {"health":"Okay"}

@app.post("/generate")
@limiter.limit("10/minute")
async def Generation(
    body: Rag,                                         
    current_user: Annotated[dict, Depends(get_current_user)]):
    query = body.query 
    
   
    generator =Retrival(path='Documents/') 
    token_stream = generator.retrieve_stream(query) 
    return StreamingResponse(token_stream, media_type="text/plain")

@app.post("/Signin")
@limiter.limit("5/minute")
def Signin(auth:Auth):
    new_user=signin(auth).check_existing_user()
    if new_user:
        return signin.save_newuser()
    else:
        return False

@app.post("/login")
@limiter.limit("5/minute")
def Login(auth: Auth):
    token = login(auth).generate_token()
    if token:
        return {"access_token": token, "token_type": "bearer"}
    return False
