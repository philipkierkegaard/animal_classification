# Use a lightweight Python image
FROM python:3.11-slim AS base

# Install system-level dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Cloud Run
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Copy application files
COPY src src/
COPY requirements.txt requirements.txt
COPY README.md README.md

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Cloud Run will use $PORT)
EXPOSE $PORT

# Start the application with a dynamic port
CMD ["uvicorn", "src.animal_classification.api:app", "--host", "0.0.0.0", "--port", "$PORT"]
