import numpy as np
import os
import argparse

def shard_dataset(input_file, output_dir, shard_size=1000000):
    """
    Splits a large numpy array file into smaller shards.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_file}...")
    data = np.load(input_file, mmap_mode='r')
    total_len = len(data)
    num_shards = (total_len + shard_size - 1) // shard_size
    
    print(f"Splitting into {num_shards} shards of size {shard_size}...")
    
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, total_len)
        shard = data[start:end]
        
        shard_path = os.path.join(output_dir, f"shard_{i:04d}.npy")
        np.save(shard_path, shard)
        print(f"Saved {shard_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input .npy file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for shards')
    parser.add_argument('--size', type=int, default=100000000, help='Shard size in tokens')
    args = parser.parse_args()
    
    shard_dataset(args.input, args.output, args.size)
