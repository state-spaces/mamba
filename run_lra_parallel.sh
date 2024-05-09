#!/bin/bash

# Run each script on a separate GPU
CUDA_VISIBLE_DEVICES=0 python -m ssm-benchmark-master.train --config aan-mamba-real.yaml &
#CUDA_VISIBLE_DEVICES=1 python -m ssm-benchmark-master.train --config aan-mamba-complex.yaml &
#CUDA_VISIBLE_DEVICES=2 python -m ssm-benchmark-master.train --config aan-S4-real.yaml &
#CUDA_VISIBLE_DEVICES=3 python -m ssm-benchmark-master.train --config aan-S4-complex.yaml &
#CUDA_VISIBLE_DEVICES=4 python -m ssm-benchmark-master.train --config imdb-mamba-real.yaml &
#CUDA_VISIBLE_DEVICES=5 python -m ssm-benchmark-master.train --config imdb-mamba-complex.yaml &
#CUDA_VISIBLE_DEVICES=6 python -m ssm-benchmark-master.train --config imdb-S4-real.yaml &
#CUDA_VISIBLE_DEVICES=7 python -m ssm-benchmark-master.train --config imdb-S4-complex.yaml &

wait
