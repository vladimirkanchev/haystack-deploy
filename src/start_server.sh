#!/bin/bash

# Find process using port 8002 and kill it
pid=$(lsof -t -i:8002)
if [ -n "$pid" ]; then
    kill -9 $pid
fi

# Start the FastAPI server
uvicorn main_deploy:app --host 127.0.0.1 --port 8002