# Use a slim Python base image
FROM python:3.13.5-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# DEBIAN_FRONTEND is set to noninteractive to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Manual Ollama Installation ---
# This is the new, robust method.
# Download the latest Linux binary from Ollama's GitHub releases
# The URL below is a generic way to get the latest release
RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz \
    -o /tmp/ollama-linux-amd64.tgz

# Extract the tarball and place the binary directly in /usr/local/bin/
RUN tar -xzf /tmp/ollama-linux-amd64.tgz -C /usr/local/bin/

# Remove the temporary file
RUN rm /tmp/ollama-linux-amd64.tgz

# Copy your application files
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit (useful for Hugging Face Spaces)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Entrypoint to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]