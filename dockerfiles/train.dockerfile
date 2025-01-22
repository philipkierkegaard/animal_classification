# Base image
FROM python:3.11-slim
#RUN pip install wandb
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV WANDB_API_KEY=6ef13821cd6974d79eeb91a1119285e45597a314


COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
#COPY wandb_tester.py wandb_tester.py

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose


ENTRYPOINT ["python", "-u", "src/animal_classification/train.py"]

