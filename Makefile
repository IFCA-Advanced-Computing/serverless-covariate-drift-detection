.ONESHELL:
SHELL := /bin/bash

VENV=.venv

install:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	pip3 install --upgrade pip &&\
				 pip3 install -r api/requirements/requirements.txt \
				              -r api/requirements/tox_requirements.txt

tox:
	$(VENV)/bin/tox

serve-dev:
	uvicorn detector_api.app.main:app --host 0.0.0.0 --port 5001 --workers 1

serve-prod:
	gunicorn -b 0.0.0.0:5001 -w 1 -t 120 -c gunicorn_conf.py -k uvicorn.workers.UvicornWorker api.app.main:app

build:
	docker build -t data-drift-detection detector_api

run:
	docker run --name data-drift-detection -p 5001:8000 data-drift-detection
