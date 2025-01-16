# Elastin Research Project Management

.PHONY: all
all: help

.PHONY: help
help:
	@echo "Elastin Research Project Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup         Setup research environment"
	@echo "  make jupyter       Start Jupyter Lab"
	@echo "  make data         Process research data"
	@echo "  make test         Run research tests"
	@echo "  make monitoring   Start monitoring stack"
	@echo "  make clean        Clean environment"

.PHONY: setup jupyter data test monitoring clean
setup:
	@echo "Setting up research environment..."
	conda env create -f environment.yml
	pip install -r requirements.txt

jupyter:
	@echo "Starting Jupyter Lab..."
	docker-compose -f docker/docker-compose.yml up jupyter

data:
	@echo "Processing research data..."
	python scripts/process_data.py

test:
	@echo "Running research tests..."
	pytest tests/

monitoring:
	@echo "Starting monitoring stack..."
	docker-compose -f docker/docker-compose.yml up -d prometheus grafana

clean:
	@echo "Cleaning environment..."
	docker-compose -f docker/docker-compose.yml down -v
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf data/processed/* 