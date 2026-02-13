# Docker build + run (gw-wl-emulator)

## 1) Build the image
From repo root:

```bash
docker build -t gw-wl-emulator:py -f docker/Dockerfile .
docker run --rm -it gw-wl-emulator:py
python3 /work/python/testing_api.py
python3 -c "import gwlensing; print(gwlensing.__file__)"
docker run --rm -it -v "${PWD}:/work" gw-wl-emulator:py
