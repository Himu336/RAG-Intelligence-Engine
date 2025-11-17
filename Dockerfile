# Use Python 3.10 (compatible with pydantic-core wheels)
FROM python:3.10-slim

# Prevent Python from buffering
ENV PYTHONUNBUFFERED=1

# Create work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
