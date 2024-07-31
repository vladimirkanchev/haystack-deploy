<div align="center">
  <img src="/_media/seven_wonders.jpeg" width="800" height="500">
</div>

# Ask AI about the Seven Wonders of the Ancient World

This repository contains a toy artificial intelligence (AI) project that provides information about a given wonder of the ancient world using a large language model (LLM) and related wikipedia data. Our idea is to help aspiring historians and AI nerds to extend their knowledge about ancient history and acquire skills to use AI/LLM models with a retrieval-augmented generation (RAG) algorithm to solve similar problems using a large knowledge base.

## A List of the Seven Ancient Wonders:
The seven ancient wonders are architectural and artistic monuments in the Eastern Mediterranean and the Middle East during antiquity, which symbolize human ingenuity and architectural excellence. Unfortunately, only the Great Pyramid of Giza still exists today, while others were destroyed during late antiquity and our concept of them is based on the historical documents and memories of people living in that age. 

The list of ancient wonders includes:

- Temple of Artemis
- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Colossus of Rhodes
- Lighthouse of Alexandria 
- Mausoleum at Halicarnassus
- Statue of Zeus


Our question-answering (Q&A) system allows the user to formulate their own questions, e.g. to ask the system: *What's the Colossus of Rhodes?*, and to obtain the corresponding AI-generated answer.

<div align="center">
  <img src="/_media/seven_wonders_map.jpg" width="800" height="500">
</div>


## Technical details 

At the current moment, the project represents a Q&A system based on a RAG algorithm using a Haystack 2.0 framework [1]. The backend is provided by FastAPI, while the frontend represents a simple javascript page at the moment. The knowledge base is stored in a specialized (in-memory store at the moment) store as embedding vectors, which are used later to build the query context. The aim of the intelligent part of the Q&A system is to find out the best context of each query through distance calculation between the query embedding and all embedding vectors. Finally, the extended query (*context + query*) is sent to the LLM model in the system and its response serves as a Q&A system answer.

Actually, our Q&A system provides two functionalities: the first functionality is that of an ordinary Q&A system, where a user can ask questions and receive answers. The second functionality uses hard-coded questions with ground-truth answers for RAG algorithm evaluation and then, users can perform additional research and can extend it for other purposes. Parameters of the RAG algorithm are provided into *src/config.yaml* file and we use a preprocessed dataset of *chunks* from wikipedia for the seven wonders [2].
    
This project started as a training project based on a notebook [3] to answer questions about the seven wonders of the ancient world. Then, we developed it into a small AI system with the RAG algorithm.


## Requirements

To run the AI models in the project, you will need an OPENAI API KEY token for the commercial OpenAI model or an HF_API_TOKEN token for the open source AI model. They should be set in your local environment as follows:
```
OPENAI_API_KEY=''
```
```
HF_API_TOKEN=''
```

## Installation, Setup and Run

1. To run the project in your local system, you need to clone and install the project first:
```
git clone https://github.com/vladimirkanchev/haystack-train-rag
cd haystack-train-rag
pip install -r requirements.txt
```
2. Start your local environment (replace '.haystack-env' with your own environment):
```
source .haystack-env/bin/activate 
```
3. Before you start the Q&A program locally, run the following script to convert loaded data chunks to vector embeddings and save them into the haystack in-memory data store:
```
python src/ingest.py
```

4. You have two options to consider: run *main.py* file to see how it works, make some experiments and get the evaluation results (step 5) or simply ask the Q&A system a question you want (step 6).


5. Then run the following file to process pre-defined inquiries about certain ancient world wonders and then to obtain the corresponding system answers and their evaluations:
```
python main.py
```

6. Run the following script to ask a question about one of the seven wonders from the ancient world:
```
python app.py
```

This will start the local fastapi server, which you can access it through *localhost:8001* and finally, you can enter your question through a simple user interface (UI):

<div align="center">
  <img src="/_media/ui_fastapi_rag.jpg" width="700" height="450">
</div>

## Technologies

In this project we use the following software technologies:
    
- Visual Studio Code 1.90
- Python 3.10
- Docker 26.1

    
## Python Packages Used
    
Some of the python packages which are used for our project are the following:

- datasets 2.19
- fastapi 0.111
- haystack-ai 2.2
- huggingface_hub 0.23
- openai 1.33
- pandas 2.2
- sentence-transformers 2.3
- tokenizers 0.19
- torch 2.3
- transformers 4.41


## Results and Evaluations

At the current moment, we have implemented three RAG algorithms: *dense* (sentence transformers, *cos* distance), *sparse* (*bm25* algorithm, no-embedding), and *hybrid* (both *dense* and *sparse*) algorithms. We have used two types of LLM models: a proprietary *openai gpt-3.5* and an open-source *HuggingFaceH4/zephyr-7b-beta* model.

We have already implemented two evaluation metrics for the RAG algorithms: **faithfulness** (measures the factual consistency of the answer against the retrieved context - reverse of presence of hallucinations) and **sas_evaluator** (measures the semantic similarity between the predicted answer and the ground-truth answer using a fine-tuned language model).

We plan to extend the evaluation part using metrics from *deep-eval* [4] and *RAGAS* [5] frameworks. The issues we have here are related to the general lack of ground-truth data for questions and significant delays during the data ingestion and the Q&A system answering.


## Future Work

Our next tasks are as follows:
   
- apply a preprocessing algorithm
- apply better and more accurate RAG algorithms
- add a vector database instead of an in-memory datastore 
- add another UI - streamlit
- extend the evaluation part with other metrics


## Who wants to contribute

Contributions, issues and feature requests will be welcomed at a later stage of the project development. 


## Reference

[1] https://haystack.deepset.ai/ 

[2] https://huggingface.co/datasets/bilgeyucel/seven-wonders

[3] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

[4] https://docs.confident-ai.com/docs/guides-rag-evaluation

[5] https://docs.ragas.io/en/latest/index.html
