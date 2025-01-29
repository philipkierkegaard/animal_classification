# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision fastapi uvicorn google-cloud-storage pillow

# Expose port 8080
EXPOSE 8080

# Start FastAPI server when the container launches
CMD ["uvicorn", "src.animal_classification.model_predict:app", "--host", "0.0.0.0", "--port", "8080"]