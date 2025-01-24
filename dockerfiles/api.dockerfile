# Use an official Python runtime as a parent image
FROM python:3.11-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt to leverage Docker caching for dependencies
COPY requirements.txt .

# Install dependencies specified in requirements.txt (cached if unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install torch torchvision gradio google-cloud-storage pillow

# Copy the rest of the application files
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Define environment variables for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

# Command to run the Gradio app
CMD ["python", "src/animal_classification/model_predict.py"]
