import os
from fastapi import Header, HTTPException

API_KEY = os.environ["API_KEY"]
ADMIN_API_KEY = os.environ["ADMIN_API_KEY"]

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
