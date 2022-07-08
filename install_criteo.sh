#!/usr/bin/env bash

python3.6 -m venv .agg_model_crto_venv
. .agg_model_crto_venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements_criteo.txt
register-kernel --venv .agg_model_crto_venv --name aggmodels_crto_env
