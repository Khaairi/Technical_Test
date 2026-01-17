from fastapi import FastAPI
from routers import router

# initialize the FastAPI application
app = FastAPI(title="Learning RAG Demo")

# include the router which contains all API endpoints
app.include_router(router)