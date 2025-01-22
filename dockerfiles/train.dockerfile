# Base image
FROM python:3.11-slim

# Set up dependencies
RUN apt update && apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for W&B API key (to be provided at runtime)
ENV WANDB_API_KEY=6ef13821cd6974d79eeb91a1119285e45597a314

# Copy application files
COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml

# Install dependencies
RUN pip install wandb && \
    pip install -r requirements.txt --no-cache-dir --verbose && \
    pip install . --no-deps --no-cache-dir --verbose

# Set entrypoint for the training script
ENTRYPOINT ["python", "-u", "src/animal_classification/train.py"]

