FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONPATH=/work/build:$PYTHONPATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libgsl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

RUN pip install --no-cache-dir numpy matplotlib scipy pybind11

# Build explicitly with pybind11_DIR set
RUN chmod +x /work/scripts/build.sh && \
    export pybind11_DIR=$(python3 -m pybind11 --cmakedir) && \
    /work/scripts/build.sh

RUN ls -l /work/build/gwlensing*.so && \
    cp /work/build/gwlensing*.so \
      $(python3 -c "import site; print(site.getsitepackages()[0])")/



CMD ["bash"]
