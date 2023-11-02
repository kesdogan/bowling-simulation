init:
	pip install -r requirements.txt
	pre-commit install

format:
	black src
	isort src

lint:
	flake8 src

run:
	python src/run.py