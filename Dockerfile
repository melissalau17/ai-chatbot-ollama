# Use the official Ollama image as the base
FROM ollama/ollama

# Switch to the root user to install system and Python packages
USER root

# Install Python and pip (Ollama image doesn't have it by default)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ollama pull phi3:mini:4k-instruct-q4_K_M
RUN ollama pull deepseek-coder:1.3b

# Copy your application files
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# --- Ollama will be running on container startup ---
# The entrypoint will now start the Streamlit app
ENTRYPOINT ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]