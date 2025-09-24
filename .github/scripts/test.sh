#!/bin/bash

set -exou pipefail

pip install dist/*.whl
python -c "import mamba_ssm; print(mamba_ssm.__version__)"