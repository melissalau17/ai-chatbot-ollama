FROM python:3.13.5-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the work directory and copy your application files
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Entrypoint to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]