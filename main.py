from fastapi import FastAPI
from routers import router

app = FastAPI(title="Learning RAG Demo")

app.include_router(router)