.PHONY: install run-api run-ui ingest test docker-up docker-down lint

install:
	pip install -r requirements.txt

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/app.py

ingest:
	python scripts/ingest.py --dir ./data/sample_docs --save ./data/vectorstore

test:
	pytest tests/ -v --tb=short

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
