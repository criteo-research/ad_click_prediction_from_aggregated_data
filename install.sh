#!/usr/bin/env bash

python3.6 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
register-kernel --venv .venv --name aggregate_models