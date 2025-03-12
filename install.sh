#!/bin/sh

cd overcooked_ai
python setup.py develop
cd ..

cd baselines
python setup.py develop
cd ..