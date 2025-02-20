# Dockerfile.backend
FROM python:3.10-slim

# Install OS-level dependencies (ffmpeg, git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python libraries directly
RUN pip install --upgrade pip && pip install numpy torch transformers langchain pydantic requests 

RUN pip install   scipy \
    uvicorn \
    fastapi 
  

RUN pip install sounddevice 

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    build-essential

RUN pip install  webrtcvad
RUN pip install  aioredis
RUN pip install  librosa 

RUN pip install tritonclient[all]

RUN apt-get update && apt-get install -y \
    portaudio19-dev  
    
RUN pip install TTS 
#coqui-tts

RUN apt-get update && apt-get install -y espeak-ng

RUN pip install websockets

RUN pip install "uvicorn[standard]"

# Set working directory inside container
WORKDIR /app

# Copy all files from current directory into /app
COPY . /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    pulseaudio-utils \
    alsa-utils \
    libasound2 \
    libasound2-dev \
    libpulse-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 chris

RUN mkdir -p /home/chris && chown -R 1000:1000 /home/chris
ENV HOME=/home/chris

RUN mkdir -p /home/chris/.cache/huggingface/hub && chown -R 1000:1000 /home/chris/.cache
ENV HF_HOME=/home/chris/.cache/huggingface/hub


# Copy the entrypoint script.
#COPY entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh

# Expose FastAPI port.
EXPOSE 7000

#USER chris

# Set the entrypoint.
#ENTRYPOINT ["/entrypoint.sh"]



# Run the FastAPI application
CMD ["uvicorn", "new_fast_api:app", "--host", "0.0.0.0", "--port", "7000"]
