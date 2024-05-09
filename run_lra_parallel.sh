#!/bin/bash

# Run each script on a separate GPU
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 python -m ssm-benchmark-master.train --config aan-mamba-real.yaml &
CUDA_VISIBLE_DEVICES=1 taskset -c 11-20 python -m ssm-benchmark-master.train --config aan-mamba-complex.yaml &
CUDA_VISIBLE_DEVICES=2 taskset -c 21-30 python -m ssm-benchmark-master.train --config aan-S4-real.yaml &
CUDA_VISIBLE_DEVICES=3 taskset -c 31-40 python -m ssm-benchmark-master.train --config aan-S4-complex.yaml &
CUDA_VISIBLE_DEVICES=4 taskset -c 41-50 python -m ssm-benchmark-master.train --config imdb-mamba-real.yaml &
CUDA_VISIBLE_DEVICES=5 taskset -c 51-60 python -m ssm-benchmark-master.train --config imdb-mamba-complex.yaml &
CUDA_VISIBLE_DEVICES=6 taskset -c 61-70 python -m ssm-benchmark-master.train --config imdb-S4-real.yaml &
CUDA_VISIBLE_DEVICES=7 taskset -c 71-80 python -m ssm-benchmark-master.train --config imdb-S4-complex.yaml &

wait
