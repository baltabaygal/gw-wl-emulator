FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

# Install Python packages needed by the plotting script
RUN pip3 install --no-cache-dir numpy matplotlib

RUN chmod +x /work/scripts/build.sh && /work/scripts/build.sh

CMD ["bash"]

