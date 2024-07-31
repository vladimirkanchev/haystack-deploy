#!/bin/bash

# Activate the virtual environment (if any)
# source /path/to/your/venv/bin/activate
# /milvus-lite/milvus start &

# # Wait for Milvus server to start
# until nc -z -v -w30 localhost 19530
# do
#   echo "Waiting for Milvus server to start..."
#   sleep 5
# done
# Run the data ingestion script


# Start the first server
# /bin/bash /code/src/start_server_uvicorn.sh &

# Start the second server
/bin/bash /code/src/start_server_streamlit.sh &

# Wait for both servers to exit
wait