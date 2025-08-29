# Use a slim Python base image
FROM python:3.13.5-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Add Ollama Installation ---
# Download and install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

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