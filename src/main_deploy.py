import json

from typing import Tuple, List

import box
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn 

from fastapi.middleware.cors import CORSMiddleware
import yaml

from rag_system.responds import get_respond_fastapi

app = FastAPI(
    title="AI Q&A system for seven wonders using API",
    description = 'A simple demo',
    version='0.0.1'
)

origins=[
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
def index():
    return {"message":"Welcome to AI Q&A system for seven wonders using API"}


@app.post("/get_answer")
async def answer(question: str = Form(...)):
    """Load output result of the inference of the rag algorithm."""
    if not question:
        raise HTTPException(status_code=404)
    answer, relevant_documents = get_respond_fastapi(question)
 
    return{"answer": answer,
           "relevant_documents": relevant_documents
          }


@app.post("/get_answer_gui")
async def answer_gui(request: Request, question: str = Form(...)):
    """Load output result of the inference of the rag algorithm."""
    if not question:
        raise HTTPException(status_code=404)
    answer, relevant_documents = get_respond_fastapi(question)
    response_data = jsonable_encoder(json.dumps(
        {"answer": answer,
         "relevant_documents": relevant_documents
         }
    )
    )
    res = Response(response_data)

    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)