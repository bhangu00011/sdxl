# Use RunPod's official PyTorch base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# We use --no-cache-dir to keep the image small
# We avoid upgrading torch/torchvision as they are specialized in the base image
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code and test input
COPY handler.py .
COPY test_input.json .

# Start the handler
CMD ["python", "-u", "handler.py"]
