FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work 
COPY . /work

RUN chmod +x /work/scripts/build.sh && /work/scripts/build.sh

CMD ["bash"]

