"""Entry point for the ai/rag app deployed through fastapi server in EC2."""

from contextlib import asynccontextmanager
import json
from time import time
from typing import Optional, AsyncGenerator
import yaml

import box
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from milvus_haystack import MilvusDocumentStore
import uvicorn

from rag_system.rag_pipelines import select_rag_pipeline
from rag_system.responds import get_respond_fastapi
from rag_system.ingest import load_data_into_store

load_dotenv(find_dotenv())

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

doc_store: Optional[MilvusDocumentStore] = None
# Configure templates
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Load data into the document store at startup."""
    global doc_store
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            doc_store = load_data_into_store()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize document " +
                           f"store after multiple attempts: {e}"
                )
    yield

app = FastAPI(
    title="AI Q&A system for seven wonders using API",
    description='A simple demo',
    lifespan=lifespan,
    version='0.0.1'
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"]
)
app = FastAPI()


@app.get("/")
async def index(request: Request):
    """Load html file at start time."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    """Load output result of the inference of the rag algorithm."""
    if not doc_store:
        raise HTTPException(status_code=500,
                            detail="Document store not initialized")

    rag_pipeline = select_rag_pipeline(doc_store)
    if not question:
        raise HTTPException(status_code=404,
                            detail="Error in question:")

    try:
        answer, relevant_documents = get_respond_fastapi(str(question),
                                                         rag_pipeline)
        response_data = jsonable_encoder(json.dumps({
            "answer": answer,
            "relevant_documents": relevant_documents}))
        return Response(response_data)
    except HTTPException as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing request: {e}")


def run():
    """Start a fastapi server."""
    uvicorn.run("fastapi_aws_deploy:app",
                host="0.0.0.0",
                port=8006,
                reload=True)


if __name__ == "__main__":
    run()
