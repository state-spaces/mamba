# SSM Benchmark on Long-range Arena (LRA)

**Note:** This repo is still under development. Currently only the configs for CIFAR-10 are available, but we will add the other configs including links to some example W&B runs in the coming days.

This repository provides the implementation for the paper:

**State Space Models as Foundation Models: A Control Theoretic Overview**  
Carmen Amo Alonso\*, Jerome Sieber\*, Melanie N. Zeilinger  
Submitted to the 63rd IEEE Conference on Decision and Control (CDC), 2024.  
(\*equal contribution)  
[arXiv](https://arxiv.org/abs/2403.16899)

## References
This repository reuses code from various other repositories.
- `mamba-1.1.4/` contains the code of [mamba-ssm (release 1.1.4)](https://github.com/state-spaces/mamba/tree/v1.1.4). We include it here to facilitate on-the-fly changes to the original [Mamba architecture](https://arxiv.org/abs/2312.00752).
- `accelerated-scan-0.1.2/` contains the code of [accelerated-scan (release 0.1.2)](https://github.com/proger/accelerated-scan/tree/0.1.2), which is used to implement the Hawk models.
- The Hawk models are adapted versions of the implementation provided by [Hippogriff](https://github.com/proger/hippogriff).
- `dataloaders/` is an adapted implementation of the code provided in the [S5 repository](https://github.com/lindermanlab/S5/tree/main).

## Requirements & Installation
To run the code on your own machine, run `pip install -r requirements.txt`. Then, install `mamba-ssm` in development mode with `pip install mamba-1.1.4/`. Finally, build the `accelerated-scan` package by running `python -m build` in the `accelerated-scan-0.1.2/` folder and install it with `pip install accelerated-scan-0.1.2/`.  
(**Warning:** The `accelerated-scan` version shipped with this repo only requires `torch >= 2.0.0`, i.e., the `triton` method will not work properly.)

## Data Preparation
See the dataloaders [README](dataloaders/README.md) for more details.

## Example Usage
The configurations to run the LRA experiments from the paper are located in `configs/`. For example, to train Mamba on the LRA text task (character level IMDB), run `python train.py --config imdb-mamba.yaml`.
To log with W&B, fill in the wandb arguments in the config files.

## Repository Structure
Directories and files that ship with GitHub repo:
```
accelerated-scan-0.1.2/ Snapshot of accelerated-scan release 0.1.2, with downgraded torch requirement.
configs/                YAML configuration files for each experiment.
dataloaders/            Code mainly derived from S5 processing each LRA dataset.
mamba-1.1.4/            Snapshot of mamba-ssm release 1.1.4. If installed in development mode, change the model in here.
models/                 Model definitions of Mamba & Hawk.
requirements.txt        Requirements for running in GPU mode (installation can be highly system-dependent).
train.py                Training loop entrypoint.
```

Directories that may be created on-the-fly:
```
data/                   Default data path used by dataloaders.
wandb/                  Local WandB log files.
```

## Citation
Please use the following BibTex entry when citing our work:
```
@misc{alonso2024state,
      title={State Space Models as Foundation Models: A Control Theoretic Overview}, 
      author={Carmen Amo Alonso and Jerome Sieber and Melanie N. Zeilinger},
      year={2024},
      eprint={2403.16899},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2403.16899}
}
```

Please reach out if you have any questions.