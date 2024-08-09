<div align="center">
  <img src="/_media/seven_wonders.jpeg" width="800" height="500">
</div>

# Ask AI about the Seven Wonders of the Ancient World

Welcome to the "Ask AI about the Seven Wonders of the Ancient World" project! This repository features a question-answering (Q&A) system, based on artificial intelligence (AI). The system provides intriguing information on the Seven Wonders of the Ancient World using a large language model (LLM) and Wikipedia data. Our goal is to help aspiring historians and AI enthusiasts extend their knowledge of ancient history while gaining practical experience with LLM models. The Q&A system demonstrates how to solve various informational queries using a retrieval-augmented generation (RAG) algorithm and a large knowledge base. It is deployed with a standalone Milvus vector database in a Docker Compose environment ensuring efficient data retrieval. 

## A List of the Seven Ancient Wonders:

The Seven Wonders are renowned architectural and artistic monuments that were located in the Eastern Mediterranean and the Middle East during antiquity. They symbolize human ingenuity and architectural excellence. Unfortunately, only the Great Pyramid of Giza remains today, while others were destroyed in late antiquity. Our understanding of these wonders is based on historical documents and the accounts of people from that age. 

The list of Ancient Wonders includes:

- Temple of Artemis
- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Colossus of Rhodes
- Lighthouse of Alexandria 
- Mausoleum at Halicarnassus
- Statue of Zeus


Our Q&A system allows the user to ask their own questions, such as: *'What's the Colossus of Rhodes?'*, and receive AI-generated answer.

<div align="center">
  <img src="/_media/seven_wonders_map.jpg" width="800" height="500">
</div>


## Technical details 

At the current moment, the project represents a Q&A system based on a RAG algorithm using the Haystack 2.0 framework [1]. The backend runs on FastAPI, while the frontend is a basic JavaScript page at the moment. The knowledge base is stored as embedding vectors in a standalone Milvus[2] vector database, derived from a preprocessed Wikipedia dataset (of data *chunks*) focused on the Seven Wonders [3].

The embedded vectors are later used to build the query context. The aim of the intelligent part of the Q&A system is to find out the best context of each query through cosine distance calculation between the query embedding and all embedding vectors in the database. Finally, the extended query (*context + query*) is submitted to the LLM model, which generates the answer for the Q&A system. Currently, the system supports only the OpenAI model as the LLM model.

Our Q&A system offers two features: the first one is of a standard Q&A system, where a user can ask questions and receive answers. The second feature uses predefined questions with ground-truth answers (obtained through googling :) ) to evaluate the RAG algorithm, allowing users to perform additional research and extend the system. Parameters for the RAG algorithm in both cases are configured into *src/config.yaml* file. 
    
This project started as a training project based on a notebook [4] designed to answer questions about the Seven Wonders of the Ancient World. It has since grown into a small Q&A system using the RAG algorithm and is now being expanded into a larger system, deployed with Docker Compose. The deployment includes separate containers for the Seven Wonders app frontend/backend with the Q&A system, the Milvus vector database, the *etcd* server with key-value configuration storage, and *minio* for object storage.


## Requirements

To run the LLM models in this project, you will need an OpenAI API key. The key is necessary for accessing the commercial OpenAI model. Set the API key set in your project folder within a  **.env** file or you can define it directly as follows:

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

We have implemented pipelines for three types of RAG algorithms: *dense* (sentence transformers, *cos* distance), *sparse* (*bm25* algorithm, no-embedding), and *hybrid* (combining both *dense* and *sparse* approaches) algorithms. The current end-to-end Q&A system uses only dense algorithm. Future updates will integrate the other algorithms and transformers to further enhance performance and results.

Initially, we planned to incorporate two types of LLM models: the proprietary *OpenAI GPT-3.5* and *HuggingFaceH4/zephyr-7b-beta*. At present, only the first model is integrated, while the second one is pending. We also intend to include local LLM models, such as LLama, which are gainig popularity.

Currently, the app is designed to work on CPU machines, as the data volume and processing requirements do not necessitate use of GPU. Additionally, including GPU-optimized Torch packages would significantly increase the size of the Docker container of our app, which we aim to avoid.
 
We have implemented an evaluation metric for the RAG algorithms: **faithfulness**, which measures the factual consistency of the answer against the retrieved context (essentially assessing the absence of hallucinations). Another metric- **sas_evaluator** , which evaluates the semantic similarity between the predicted answer and ground-truth answers using a fine-tuned language model is under development. This metric requires a cross-encoder, which adds another challenge to our list of challenges such as the general lack of ground-truth data for the Q&A app questions, delays in data ingestion and answer evaluation. We plan to extend the evaluation framework by incorporating metrics and data from the *deep-eval* [5] and *RAGAS* [6] frameworks.


## Future Work

Our next tasks include:
  
- Implementing additional RAG algorithms and integrating new transformers/embeddings.
- Adding a Kubernetes deployment option which will improve scalability and manageability. 
- Dockerizing the alternative UI using Streamlit providing more options to interact with the system.
- Expanding the evaluation framework with additional metrics to better assess the performance and accuracy of the system.


## Who wants to contribute

We will welcome contributions, issues, and feature requests as the project progresses to a later development stage.


## Reference

[1] https://haystack.deepset.ai/ 

[2] https://milvus.io/docs/v2.0.x/install_standalone-docker.md

[3] https://huggingface.co/datasets/bilgeyucel/seven-wonders

[4] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

[5] https://docs.confident-ai.com/docs/guides-rag-evaluation

[6] https://docs.ragas.io/en/latest/index.html
