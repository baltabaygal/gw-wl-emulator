FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libgsl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

RUN pip install --no-cache-dir numpy matplotlib scipy pybind11

# Build explicitly with pybind11_DIR set
RUN chmod +x /work/scripts/build.sh && \
    export pybind11_DIR=$(python -m pybind11 --cmakedir) && \
    /work/scripts/build.sh

CMD ["bash"]
