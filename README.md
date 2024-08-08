<div align="center">
  <img src="/_media/seven_wonders.jpeg" width="800" height="500">
</div>

# Ask AI about the Seven Wonders of the Ancient World

Welcome to the "Ask AI about the Seven Wonders of the Ancient World" project! This repository features a toy artificial intelligence (AI) system0 that provides intriguing information on the Seven Wonders of the Ancient World using a large language model (LLM) and Wikipedia data. Our idea is to help aspiring historians and AI enthusiasts to extend their knowledge about ancient history and acquire practical experience with AI/LLM models. The system demonstrates how to solve varied informational queries using a retrieval-augmented generation (RAG) algorithm and a large knowledge base.The AI RAG system is deployed through standalone Milvus vector database in a Docker Compose environment which ensures efficient data retrieval. 

## A List of the Seven Ancient Wonders:

The Seven Wonders are renowned architectural and artistic monuments located in the Eastern Mediterranean and the Middle East during antiquity. They symbolize human ingenuity and architectural excellence. Unfortunately, only the Great Pyramid of Giza remains today, while others were destroyed during late antiquity. Our understanding of them is based on the historical documents and memories of people from that age. 

The list of ancient wonders includes:

- Temple of Artemis
- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Colossus of Rhodes
- Lighthouse of Alexandria 
- Mausoleum at Halicarnassus
- Statue of Zeus


Our question-answering (Q&A) system allows the user to ask their own questions, such as: *What's the Colossus of Rhodes?*, and receive the corresponding AI-generated answer.

<div align="center">
  <img src="/_media/seven_wonders_map.jpg" width="800" height="500">
</div>


## Technical details 

At the current moment, the project represents a Q&A system based on a RAG algorithm using the Haystack 2.0 framework [1]. The backend is provided by FastAPI, while the frontend represents a simple JavaScript page at the moment. The knowledge base is stored in a standalone Milvus[2] vector database as embedding vectors loaded from a Wikipedia dataset - a preprocessed dataset of *chunks* from Wikipedia focused on the Seven Wonders [3].

The embedded vectors are later used to build the query context. The aim of the intelligent part of the Q&A system is to find out the best context of each query through cosine distance calculation between the query embedding and all embedding vectors in the database. Finally, the extended query (*context + query*) is sent to the LLM model in the system and its response serves as the Q&A system answer. Currently, only the OpenAI model is supported as LLM model.

Our Q&A system provides two functionalities: the first is that of a standard Q&A system, where a user can ask questions and receive answers. The second functionality uses hard-coded questions with ground-truth answers for RAG algorithm evaluation, allowing users to perform additional research and extend it for other purposes. Parameters of the RAG algorithm in both cases are provided into *src/config.yaml* file. 
    
This project began as a training project based on a notebook [4] to answer questions about the Seven Wonders of the Ancient World. It was subsequently developed into a small AI system utilizing the RAG algorithm and is now being expanded into a larger AI/RAG system, deployed using Docker Compose. The deployment includes separate containers for the Seven Wonders app frontend/backend with the RAG system, the Milvus vector database, the *etcd* server with key-value configuration storage, and *minio* for object storage.


## Requirements

To run the AI/LLM models in this project, you will need an OPENAI API key. The key is required for accessing the commercial OpenAI model. Set the API key set in your project folder within a  **.env** file or explicitly define it as follows:

```
OPENAI_API_KEY=''
```


## Installation, Setup and Run

To install and run the Seven Wonders app, follow these steps:

**1. Clone the repository** 

Clone the *dev* branch of the project repository to your local machine or cloud instance:

```
git clone --branch dev https://github.com/vladimirkanchev/haystack-deploy
cd haystack-deploy
```

**2. Set the OpenAI API Key**  

Add your OpenAI API key in your preferred method (e.g., in a .env file or directly in the environment).

**3.Build the Docker Container**

Build the docker container of the Seven Wonders AI/RAG app:

``` 
docker build -t seven-wonders .  
```

**4. Install the Docker Compose**

Download and install Docker Compose:

```
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

```
```
chmod +x /usr/local/bin/docker-compose
``` 

**5. Start the Docker Compose**

Launch the Seven Wonders application using Docker Compose - all containers start:

```
docker compose up
```

**6. Access the Application** 

You can enter your questions through a simple user interface (UI). If running locally, access it via *localhost:8006*. If running on a cloud instance, use the public address of your instance on port *8006*.

<div align="center">
  <img src="/_media/ui_fastapi_rag.jpg" width="700" height="450">
</div>

## Technologies

This project utilizes the following software technologies:
    
**- Visual Studio Code:** Version 1.92

**- Python:** Version 3.11.9

**- Docker:** Version 27.1

**- FastAPI:** Version 0.111

**- Streamlit:** Version 1.37

**- Haystack:** Version 2.3

    
## Python Packages Used
    
Our project relies on some the following packages:

**- datasets** Version 2.20 

**- milvus-haystack** Version 0.0.9

**- openai** Version 1.37

**- pandas** Version 2.2

**- pymilvus** Version 2.4

**- sentence-transformers** Version 2.3

**- streamlit** Version 1.37

**- tokenizers** Version 0.15

**- torch** Version 2.1

**- transformers** Version 4.39


## Results and Evaluations

Currently, we have implemented pipelines for three types of RAG algorithms: *dense* (sentence transformers, *cos* distance), *sparse* (*bm25* algorithm, no-embedding), and *hybrid* (both *dense* and *sparse*) algorithms. The current end-to-end AI/RAG system uses only dense algorithm. Future updates will integrate the other pipelines and transformers to enhance performance and results.

We initally planned to incorporate two types of LLM models: the proprietary *OpenAI GPT-3.5* and *HuggingFaceH4/zephyr-7b-beta*. At present, only the first model is integrated, while the second one is pending. We also aim to include local LLM models such  as LLama, which are gainig popularity.

Currently, the app is designed to work on CPU machines, as the data volume and processing requirements do not necessitate use of GPU. Additionally, GPU-optimized Torch packages would significantly increase the size of the Docker container of our app, which we aim to avoid.
 
We have implemented an evaluation metric for the RAG algorithms: **faithfulness**, which measures the factual consistency of the answer against the retrieved context (reverse of presence of hallucinations). Another metric- **sas_evaluator** ., which evaluates the semantic similarity between the predicted answer and the ground-truth answer using a fine-tuned language model is under development. It requires a cross-encoder, which another challenge to our list of challenges such as the general lack of ground-truth data for the Q&A app questions, delays during the data ingestion and answer evaluation. We plan to extend the evaluation framework by incorporating metrics and data from the *deep-eval* [5] and *RAGAS* [6] frameworks.


## Future Work

Our next tasks include:
  
- Implementing additional RAG algorithms and integrating new transformers/embeddings.
- Adding a Kubernetes deployment option for the Seven Wonders app. 
- Dockerizing the alternative UI using Streamlit.
- Expanding the evaluation framework with additional metrics.


## Who wants to contribute

We will welcome contributions, issues, and feature requests as the project progresses to a later development stage.


## Reference

[1] https://haystack.deepset.ai/ 

[2] https://milvus.io/docs/v2.0.x/install_standalone-docker.md

[3] https://huggingface.co/datasets/bilgeyucel/seven-wonders

[4] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

[5] https://docs.confident-ai.com/docs/guides-rag-evaluation

[6] https://docs.ragas.io/en/latest/index.html
