#!/bin/bash

source venv/bin/activate

PYTHON="venv/bin/python"
echo "venv ${PYTHON}"
echo ""

python app.py $* --disable_faster_whisper

deactivate

