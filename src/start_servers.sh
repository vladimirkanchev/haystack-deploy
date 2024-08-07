#!/bin/bash

#!/bin/bash

# Activate the virtual environment (if any)
# source .rag-deploy-env/bin/activate
# /milvus-lite/milvus start &

# Function to handle cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $FASTAPI_PID 
    #kill $STREAMLIT_PID
}

# Trap signals to ensure cleanup is called on script exit
trap cleanup EXIT

echo "Starting FastAPI server..."
uvicorn rag_system.app_fastapi:app --host 0.0.0.0 --port 8006 &
FASTAPI_PID=$!
echo "FastAPI server started with PID $FASTAPI_PID"
wait $FASTAPI_PID

# Start Streamlit App in the background
# echo "Starting Streamlit app..."
# streamlit run rag_system/app_streamlit.py --server.port 8007 &
# STREAMLIT_PID=$!
# echo "Streamlit app started with PID $STREAMLIT_PID"
# wait $STREAMLIT_PID