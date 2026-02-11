# Fast ML Emulator for GW Weak Lensing PDFs

This project aims to build a fast machine-learning emulator for weak gravitational lensing of gravitational waves, based on an existing C++ Monte-Carlo implementation from a published paper.


## Current status:
We are in the baseline C++ replication phase.
No ML components have been added yet.


# Build the Docker image

From project root:

docker build -t gw-wl-dev -f docker/Dockerfile .