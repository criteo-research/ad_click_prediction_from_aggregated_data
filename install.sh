#!/usr/bin/env bash

python3.6 -m venv .agg_model_venv
. .agg_model_venv/bin/activate
pip install -U pip
pip install -e .
pip install -r dev-requirements.txt
pip install -r tests-requirements.txt
register-kernel --venv .agg_model_venv --name aggregate_models_env
