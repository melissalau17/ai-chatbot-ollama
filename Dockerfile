FROM python:3.13.5-slim

# Install system dependencies for ctransformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the model files
WORKDIR /app
RUN mkdir -p /app/models

# Download a tiny Llama model during the build
RUN curl -L https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
    -o /app/models/llama-2-7b-chat.gguf

# Download a tiny Deepseek model during the build
RUN curl -L https://huggingface.co/TheBloke/deepseek-coder-1.3B-base-GGUF/resolve/main/deepseek-coder-1.3b-base-q4_K_M.gguf \
    -o /app/models/deepseek-coder-1.3b.gguf

# Copy your application files
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Entrypoint to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]