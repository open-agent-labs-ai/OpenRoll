.PHONY: help install test validate clean

help:
	@echo "OpenRoll — common targets:"
	@echo "  make install   - editable install of the python SDK with dev extras"
	@echo "  make test      - run the test suite"
	@echo "  make validate  - validate every bundled example"
	@echo "  make clean     - remove build/test caches"

install:
	pip install -e python/[dev]

test:
	PYTHONPATH=python python3 -m pytest python/tests/ -q

validate:
	PYTHONPATH=python python3 -m openroll.cli validate examples/

clean:
	rm -rf python/.pytest_cache python/build python/dist python/*.egg-info
	find python -name __pycache__ -exec rm -rf {} +
