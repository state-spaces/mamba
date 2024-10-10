torchrun --nproc-per-node "$1" run_dist_test.py --iterations 4 --nproc_per_node "$1" --batch_size 8 --random_seed 42
