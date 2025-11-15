FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    net-tools \
    iputils-ping \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create directories for experiments
RUN mkdir -p /workspace/experiments/logs /workspace/experiments/configs

# Expose ports for distributed training and monitoring
# 29500: PyTorch DDP master port
# 8501: Streamlit dashboard
# 6379: Ray head node (optional)
# 8265: Ray dashboard (optional)
EXPOSE 29500 8501 6379 8265

# Default command
CMD ["/bin/bash"]
