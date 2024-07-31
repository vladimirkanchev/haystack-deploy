"""Application file for fastapi endpoint for the rag algorithm."""
import json

import box
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
import uvicorn
import yaml

from rag_system.rag_pipelines import select_rag_pipeline
from rag_system.responds import get_respond_fastapi
from rag_system.ingest import load_data_into_store

load_dotenv(find_dotenv())

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


app = FastAPI()
# Configure templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    """Load html file at start time."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(str)):
    """Load output result of the inference of the rag algorithm."""
    doc_store = load_data_into_store()

    rag_pipeline = select_rag_pipeline(doc_store)
    if not question:
        raise HTTPException(status_code=404)
    print(question)
    answer, relevant_documents = get_respond_fastapi(str(question),
                                                     rag_pipeline)
    response_data = jsonable_encoder(json.dumps(
        {"answer": answer,
         "relevant_documents": relevant_documents
         }
    )
    )
    res = Response(response_data)

    return res


def run():
    """Start a fastapi server."""
    uvicorn.run("rag_system.app_fastapi:app",
                host='0.0.0.0',
                port=8004,
                reload=True)


if __name__ == "__main__":
    run()
