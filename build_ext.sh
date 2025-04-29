#!/bin/bash

set -e
pdm run build-ext-ref
cp src/extensions_ref/build/lib/tiny_llm_ext_ref/tiny_llm_ext_ref.metallib .venv/lib/python3.12/site-packages/mlx/lib/
