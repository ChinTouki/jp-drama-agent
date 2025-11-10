from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "JP Drama Agent API is running."}
