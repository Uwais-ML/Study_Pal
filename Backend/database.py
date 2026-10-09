import sqlite3
import os
db_filename="User_auth.db"
class DATABASE:
    def __init__(self,db_name:str):
        self.conn = sqlite3.connect(db_name+".db")
        self.cursor = self.conn.cursor()

    def initatedb(self):
        if os.path.exists(db_filename):
            print("Database file exists!")
            return True
        else:
            print("Database file does not exist yet.")


    def command(self,query:str,params:str):
        try:
            self.cursor.execute(query, params)
            self.conn.commit() 
            result = self.cursor.fetchone()
            
            return result[0] if result else None
        except Exception as e:
            print(e)
    def close(self):
        """Call this explicitly only when your app completely shuts down."""
        self.conn.close()

            



        