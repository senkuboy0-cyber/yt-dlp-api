FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg is required for yt-dlp to merge video+audio streams if needed, and for some extractions
# git is often useful for yt-dlp updates or specific extractors
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
# We use root user by default in this image, which allows 'pip install -U yt-dlp' to work.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
