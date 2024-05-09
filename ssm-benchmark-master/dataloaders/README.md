# Overview

Basic datasets including MNIST and CIFAR will auto-download. Source code for these datamodules are in [basic.py](basic.py).

By default, data is downloaded to `data/` (in the top-level folder of this repo) by default.

## Advanced Usage

After downloading and preparing data, the paths can be configured in several ways.

1. Suppose that it is desired to download all data to a different folder, for example a different disk.
The data path can be configured by setting the environment variable `DATA_PATH`.

2. For fine-grained control over the path of a particular dataset, set `dataset.data_dir` in the config. For example, if the LRA ListOps files are located in `/home/lra/listops-1000/` instead of the default `./data/listops/`,
pass in `+dataset.data_dir=/home/lra/listops-1000` on the command line or modify the config file directly.

3. As a simple workaround, softlinks can be set, e.g. `ln -s /home/lra/listops-1000 ./data/listops`

# Data Preparation

[LRA](#long-range-arena-lra) must be manually downloaded.

By default, these should go under `$DATA_PATH/`, which defaults to `./data`.  For the remainder of this README, these are used interchangeably.

## Long Range Arena (LRA)

LRA can be downloaded from the [GitHub page](https://github.com/google-research/long-range-arena).
These datasets should be organized as follows:
```
$DATA_PATH/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
```
The other two datasets in the suite ("Image" or grayscale sequential CIFAR-10; "Text" or char-level IMDB sentiment classification) are both auto-downloaded.