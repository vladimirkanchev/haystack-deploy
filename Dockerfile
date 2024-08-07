FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    build-essential \
    libpq-dev \
    docker.io \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip[full]
# code to code directory
COPY . /code

# Ensure start scripts are executable
RUN chmod +x /code/src/start_servers.sh
RUN chmod +x /code/src/cleanup.sh

WORKDIR /code/src 

ARG PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/code/src"
RUN pip install --no-cache-dir --upgrade -r /code/src/requirements.txt

EXPOSE 8006
EXPOSE 8007
EXPOSE 19530
EXPOSE 19531
# Install the src package in editable mode
RUN pip install -e /code/src/

ENTRYPOINT ["/code/src/start_servers.sh"]