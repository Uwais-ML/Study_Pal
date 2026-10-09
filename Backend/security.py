import bcrypt 
from config import Auth
from database import DATABASE
import datetime
import jwt
from dotenv import load_dotenv
import os
load_dotenv()
import jwt
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

class login(Auth):
    def __init__(self):
        self.username=Auth.username
        self.password=Auth.password
        self.db=DATABASE("User_auth")


    def check_existing_user(self):
        query = "SELECT password_hash FROM users WHERE username = ?"
        hash=self.db.command(query,(self.username,))
        return hash

    
    def verification(self):
        self.hash=self.check_existing_user()
        if not self.hash:
            return False
        return bcrypt.checkpw(
            self.password.encode('utf-8'), 
            self.hash.encode('utf-8')
        )

    
    def generate_token(self):
        if not self.verification():
            return None
        payload = {
            "sub": self.username, 
            "iat": datetime.datetime.now(datetime.timezone.utc),
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return token

    
class signin(Auth):


    def __init__(self):
        self.username=Auth.username
        self.password=Auth.password
        self.db=DATABASE("User_auth")


    def check_existing_user(self):
        query = "SELECT password_hash FROM users WHERE username = ?"
        hash=self.db.command(query,(self.username,))
        return hash

    
    def save_newuser(self):
        self.hash=self.check_existing_user()
        if self.hash:
            return False
        password_bytes = self.password.encode('utf-8')
        hashed_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
        hashed_string = hashed_bytes.decode('utf-8')
        query = "INSERT INTO users (username, password_hash) VALUES (?, ?)"
        self.db.command(query, (self.username, hashed_string))
        print(f"User '{self.username}' registered successfully!")
        return True

