# Dockerfile
FROM python:3.11-slim

# 1) Install OS deps so Poetry and your wheels can build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      build-essential \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Create app directory
WORKDIR /app

# 3) Copy only poetry files and install dependencies
RUN pip install poetry
COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

# 4) Copy the rest of your code
COPY . .

# 5) Make sure your run script is executable
RUN chmod +x run_all.sh

# 6) Default entrypoint
ENTRYPOINT ["bash", "run_all.sh"]
