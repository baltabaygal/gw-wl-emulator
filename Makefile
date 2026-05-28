.PHONY: help build test watch clean pytest validate-dataset generate-small-dataset install-dev

help:
	@echo "In-container targets:"
	@echo "  build  - build C++ + pybind module (./scripts/build.sh)"
	@echo "  test   - run python/testing_api.py"
	@echo "  watch  - auto-rebuild on cpp/ changes"
	@echo "  clean  - remove build"

build:
	./scripts/build.sh

test:
	python3 python/testing_api.py

pytest:
	python3 -m pytest tests

validate-dataset:
	python3 python/validate_dataset.py

generate-small-dataset:
	python3 python/generate_dataset.py --num_points 10 --nsamples 1000 --seed 123

watch:
	./scripts/watch_build.sh

clean:
	rm -rf build

install-dev:
	python3 -m pip install -r requirements.txt
