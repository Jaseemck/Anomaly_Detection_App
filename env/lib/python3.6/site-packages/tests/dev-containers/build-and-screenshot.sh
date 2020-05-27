#!/bin/bash
export PYDECK_VERSION="0.3.1"
export PYPI_INSTALL_URL=http://localhost:8080/
docker-compose build --force-rm --no-cache --parallel && docker-compose up --no-build -d
python snap.py
docker-compose down
