FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app
COPY entry.py .
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p data/repos data/processed data/models

# Expose the port
EXPOSE 5000

# Run the application with host set to 0.0.0.0 to make it accessible outside the container
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]