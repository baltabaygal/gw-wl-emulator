$IMAGE = "gw-wl-emulator:py"

# Build image if not exists
if (-not (docker images -q $IMAGE)) {
    Write-Host "Docker image not found. Building..."
    docker build -t $IMAGE -f docker/Dockerfile .
}

Write-Host "Starting development shell..."

docker run --rm -it `
    -v "${PWD}:/work" `
    -w /work `
    $IMAGE `
    bash
