
---

### `docs/dev_workflow.md` (copy/paste)
```md
# Dev workflow

There are two modes:

## A) Reproducible build (recommended for “it just works”)
Use the Dockerfile build:

```bash
docker build -t gw-wl-emulator:py -f docker/Dockerfile .
docker run --rm -it gw-wl-emulator:py
