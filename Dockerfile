# Use Python 3.12 slim image
FROM python:3.12-slim

# Prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=True

# Install system dependencies for audio + streamlit
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run requires listening on port 8080
EXPOSE 8080

# Streamlit needs this env var for base URL
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
