"""Entry point for the ai/rag app deployed through fastapi server in EC2."""
import json
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uvicorn

from rag_system.rag_pipelines import select_rag_pipeline
from rag_system.responds import get_respond_fastapi
from rag_system.ingest import load_data_into_store

app = FastAPI(
    title="AI Q&A system for seven wonders using API",
    description='A simple demo',
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

load_dotenv(find_dotenv())


app = FastAPI()
# Configure templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
def index() -> Dict[str, str]:
    """Test respond."""
    return {"message": "Welcome to AI Q&A system for seven wonders using API"}


@app.post("/get_answer")
async def answer(question: str = Form(...)) -> Dict[str, str | List[str]]:
    """Load output result of the inference of the rag algorithm."""
    doc_store = load_data_into_store()
    rag_pipeline = select_rag_pipeline(doc_store)
    print(question)
    if not question:
        raise HTTPException(status_code=404)
    curr_answer, relevant_documents = get_respond_fastapi(question,
                                                          rag_pipeline)
    k = len(relevant_documents)

    return {"answer": curr_answer,
            "relevant_documents": k
            }


@app.post("/get_answer_gui")
async def answer_gui(request: Request, question: str = Form(...)) -> Response:
    """Load output result of the inference of the rag algorithm."""
    doc_store = load_data_into_store()
    rag_pipeline = select_rag_pipeline(doc_store)
    if not question:
        raise HTTPException(status_code=404)
    curr_answer, relevant_documents = get_respond_fastapi(question,
                                                          rag_pipeline)
    response_data = jsonable_encoder(json.dumps(
        {"answer": curr_answer,
         "relevant_documents": relevant_documents
         }
    )
    )
    res = Response(response_data)

    return res

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
