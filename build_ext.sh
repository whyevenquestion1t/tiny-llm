#!/bin/bash

cd src/extensions_ref
pdm run python setup.py build_ext --inplace
