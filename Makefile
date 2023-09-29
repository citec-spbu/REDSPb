check:
	black --line-length 120 -S --check .
	isort . --check --multi-line 3  --profile black --py 38
	flake8 . --config .flake8

format:
	black --line-length 120 -S .
	isort . --multi-line 3 --profile black --py 38
