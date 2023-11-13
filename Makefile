init:
	pip install -r requirements.txt
	pre-commit install

format:
	black src
	isort src

lint:
	flake8 src

test:
	python -m pytest

run:
	python src/run.py