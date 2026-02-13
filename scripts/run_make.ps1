param(
  [Parameter(Mandatory=$false)]
  [string]$Target = "help"
)

$IMAGE = "gw-wl-emulator:py"

function Build-Image {
  docker build -t $IMAGE -f docker/Dockerfile .
}

function Shell {
  docker run --rm -it `
    -v "${PWD}:/work" `
    -w /work `
    $IMAGE `
    bash
}

# Host-side targets (need Docker)
if ($Target -eq "image") {
  Build-Image
  exit 0
}

if ($Target -eq "shell") {
  Shell
  exit 0
}

# Everything else runs INSIDE the container
docker run --rm -it `
  -v "${PWD}:/work" `
  -w /work `
  $IMAGE `
  bash -lc "make $Target"
