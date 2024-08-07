#!/bin/bash

# Stopping and removing containers
docker-compose down

# Optionally, remove any unnamed or dangling volumes
docker volume prune -f
