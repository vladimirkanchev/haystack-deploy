"""Application file for fastapi endpoint for the rag algorithm."""
from contextlib import asynccontextmanager
import json
from typing import Optional, AsyncGenerator

import box
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from milvus_haystack import MilvusDocumentStore
import uvicorn
import yaml

from rag_system.rag_pipelines import select_rag_pipeline
from rag_system.responds import get_respond_fastapi
from rag_system.ingest import load_data_into_store


load_dotenv(find_dotenv())

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Configure templates
templates = Jinja2Templates(directory="templates")
doc_store: Optional[MilvusDocumentStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Load data into the document store at startup."""
    global doc_store
    doc_store = load_data_into_store()
    yield
    # Any teardown code can go here

app = FastAPI(lifespan=lifespan)


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
        raise HTTPException(status_code=404)

    answer, relevant_documents = get_respond_fastapi(str(question),
                                                     rag_pipeline)
    response_data = jsonable_encoder(json.dumps(
        {"answer": answer,
         "relevant_documents": relevant_documents
         }
    ))
    res = Response(response_data)
    return res


def run():
    """Start a fastapi server."""
    uvicorn.run("rag_system.app_fastapi:app",
                host='0.0.0.0',
                port=8006,
                reload=True)


if __name__ == "__main__":
    run()
