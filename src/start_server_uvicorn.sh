#!/bin/bash

# Find process using port 8002 and kill it
pid=$(lsof -t -i:8004)
pid2=$(lsof -t -i:19530)
if [ -n "$pid" ]; then
    kill -9 $pid
    kill -9 $pid2
fi

# Start the FastAPI server
uvicorn main_deploy:app --host 0.0.0.0 --port 8004