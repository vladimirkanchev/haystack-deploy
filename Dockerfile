FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# code to code directory
COPY . /code

# set permissions

RUN chmod +x /code/src
RUN pip install --no-cache-dir --upgrade -r code/src/requirements.txt

EXPOSE 8004
EXPOSE 8005

WORKDIR /code/src

ENV PYTHONPATH "${PYTHONPATH}:/code/src"

CMD pip install -e src/.