# Base image with Python
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with hot reload disabled for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
