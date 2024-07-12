#!/bin/bash

# Specify the port number
PORT=8005

# Find the process using the specified port
pid=$(lsof -t -i:$PORT)
if [ -n "$pid" ]; then
    # Check if the process is related to Streamlit
    if ps -p $pid -o cmd= | grep -q "streamlit"; then
        echo "Port $PORT is in use by Streamlit process $pid. Terminating process..."
        kill -9 $pid
        echo "Process $pid terminated."
    else
        echo "Port $PORT is in use by another process. Not terminating it."
        exit 1
    fi
else
    echo "Port $PORT is not in use."
fi

# Start the Streamlit server
echo "Starting Streamlit server on port $PORT..."
python -c "
import sys
import streamlit.web.cli as stcli

def main():
    sys.argv = ['streamlit', 'run', 'rag_system/app_streamlit.py', '--server.port', '$PORT']
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
"
