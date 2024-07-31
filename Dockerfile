FROM python:3.10-slim

RUN apt-get update && apt-get install -y\
    build-essential\
    libpq-dev\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip[full]
# code to code directory
COPY . /code

# set permissions
# Ensure start scripts are executable
RUN chmod +x /code/src/start_servers.sh
RUN chmod +x /code/src/start_server_streamlit.sh
RUN chmod +x /code/src/start_server_uvicorn.sh

WORKDIR /code/src 

ARG PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/code/src"
RUN pip install --no-cache-dir --upgrade -r /code/src/requirements.txt



EXPOSE 8000
EXPOSE 8004
EXPOSE 8005
EXPOSE 19530



# Install the src package in editable mode
RUN pip install -e /code/src/

# Start the servers when the container starts
CMD ["/bin/bash", "/code/src/start_servers.sh"]
