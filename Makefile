.PHONY: serve test check
serve:
	python -m uvicorn cashwatch.main:app_factory --factory --host 127.0.0.1 --port 8000 --workers 1
test:
	python -m pytest -q
check:
	python scripts/check_model.py --model-dir models/Qwen3-VL-2B-Instruct

