# Use Python 3.11 (better compatibility with latest packages)
FROM python:3.11-slim

# Prevent Python from buffering
ENV PYTHONUNBUFFERED=1

# Create work directory
WORKDIR /app

# Install system dependencies including ffmpeg (required for Whisper and pydub)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
