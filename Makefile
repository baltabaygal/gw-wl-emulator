.PHONY: help build test watch clean

help:
	@echo "In-container targets:"
	@echo "  build  - build C++ + pybind module (./scripts/build.sh)"
	@echo "  test   - run python/testing_api.py"
	@echo "  watch  - auto-rebuild on cpp/ changes"
	@echo "  clean  - remove /work/build"

build:
	./scripts/build.sh

test:
	python3 /work/python/testing_api.py

watch:
	./scripts/watch_build.sh

clean:
	rm -rf /work/build
