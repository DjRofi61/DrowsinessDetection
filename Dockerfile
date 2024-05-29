FROM python:3.8-slim

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    alsa-utils \
    libsdl1.2-dev \
    libsdl-mixer1.2 \
    libsdl2-dev \
    libsdl2-mixer-dev \
    pulseaudio

# Create a directory for the app
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for dummy audio
ENV SDL_AUDIODRIVER=dummy

# Start the application
CMD ["python", "app.py"]
