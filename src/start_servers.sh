#!/bin/bash

# Activate the virtual environment (if any)
# source /path/to/your/venv/bin/activate

# Start the first server
/bin/bash /code/src/start_server_uvicorn.sh &

# Start the second server
/bin/bash /code/src/start_server_streamlit.sh &

# Wait for both servers to exit
wait