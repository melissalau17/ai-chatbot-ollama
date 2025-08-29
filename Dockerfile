# Use a base image that supports system-level installations
FROM python:3.13.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# git is needed if you have private repositories or need to clone things
# build-essential for compiling some Python packages
# curl for downloading Ollama
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama Installation ---
# Download and install Ollama
# Ensure the Ollama binary is available in the PATH for easy execution
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set Ollama server to listen on all interfaces (0.0.0.0)
# and use the default port (11434)
ENV OLLAMA_HOST=0.0.0.0:11434

# Expose Ollama's default port (optional, but good practice for clarity)
EXPOSE 11434

# Download an Ollama model during the build process
# You can change 'llama2' to any other model you want (e.g., 'deepseek-coder')
# This makes sure the model is available when the container starts
RUN ollama pull llama2
# If you need another model, add another 'ollama pull' command
# RUN ollama pull deepseek-coder

# --- Application Setup ---
# Copy your requirements.txt and application source code
COPY requirements.txt ./
COPY app.py

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Entrypoint: Start Ollama and Streamlit ---
# Use a custom entrypoint script to start both services
# Create a simple shell script to run both Ollama and Streamlit
# We use 'bash -c' to run multiple commands in a single entrypoint
# Ollama runs in the background, Streamlit runs in the foreground
ENTRYPOINT ["bash", "-c", "ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck for Streamlit (useful for Hugging Face Spaces)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health