.PHONY: test lint install run-npu-service run-code-download clean

install:
	pip install -r requirements.txt

lint:
	ruff check .
	mypy framework/ pipelines/ npu_service/ code_download/

test:
	pytest tests/ -v --tb=short

test-e2e:
	pytest tests/e2e/ -v --tb=short

# Start the NPU inference service locally (mock mode)
run-npu-service:
	uvicorn npu_service.server:app --host 0.0.0.0 --port 8080 --reload

# Start the code download service locally
run-code-download:
	uvicorn code_download.service:app --host 0.0.0.0 --port 8081 --reload

# Run a pipeline end-to-end in dev mode with small dataset
run-dev:
	python -m framework.bootstrap \
		--pipeline llm-distill \
		--version v2.3.1 \
		--input-path data/sample/ \
		--output-path data/output/ \
		--dev-mode

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
