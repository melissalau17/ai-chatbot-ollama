# Use a base image that supports system-level installations
FROM python:3.13.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama Installation ---
RUN curl -fsSL https://ollama.com/install.sh | sh

ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
RUN ollama pull llama2
# If you need another model, add another 'ollama pull' command
# RUN ollama pull deepseek-coder

# --- Application Setup ---
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Entrypoint & Command ---
# Start the Ollama server in the background
CMD ollama serve &

# Add a simple wait script to check if Ollama is ready
# before starting the Streamlit app
RUN echo '#!/bin/bash' > wait-for-ollama.sh && \
    echo 'until curl --fail http://localhost:11434; do echo waiting for Ollama...; sleep 1; done' >> wait-for-ollama.sh && \
    echo 'streamlit run app.py --server.port=8501 --server.address=0.0.0.0' >> wait-for-ollama.sh && \
    chmod +x wait-for-ollama.sh

# Run the wait script and start the Streamlit app
ENTRYPOINT ["./wait-for-ollama.sh"]

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health