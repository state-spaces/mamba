""" Datasets for core experimental results """

import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
from .utils import is_list, permutations
from torch.nn import functional as F

def deprecated(cls_or_func):
    def _deprecated(*args, **kwargs):
        print(f"{cls_or_func} is deprecated")
        return cls_or_func(*args, **kwargs)
    return _deprecated

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()

class DefaultCollateMixin:
    """Controls collating in the DataLoader

    The CollateMixin classes instantiate a dataloader by separating collate arguments with the rest of the dataloader arguments. Instantiations of this class should modify the callback functions as desired, and modify the collate_args list. The class then defines a _dataloader() method which takes in a DataLoader constructor and arguments, constructs a collate_fn based on the collate_args, and passes the rest of the arguments into the constructor.
    """

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(cls._collate_arg_names), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {k: loader_args[k] for k in loader_args if k in self.collate_args}
        loader_args = {k: loader_args[k] for k in loader_args if k not in self.collate_args}
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


class SequenceResolutionCollateMixin(DefaultCollateMixin):
    """self.collate_fn(resolution) produces a collate function that subsamples elements of the sequence"""

    @classmethod
    def _collate_callback(cls, x, resolution=None):
        if resolution is None:
            pass
        else:
            # Assume x is (B, L_0, L_1, ..., L_k, C) for x.ndim > 2 and (B, L) for x.ndim = 2
            assert x.ndim >= 2
            n_resaxes = max(1, x.ndim - 2) # [AG 22/07/02] this line looks suspicious... are there cases with 2 axes?
            # rearrange: b (l_0 res_0) (l_1 res_1) ... (l_k res_k) ... -> res_0 res_1 .. res_k b l_0 l_1 ...
            lhs = "b " + " ".join([f"(l{i} res{i})" for i in range(n_resaxes)]) + " ..."
            rhs = " ".join([f"res{i}" for i in range(n_resaxes)]) + " b " + " ".join([f"l{i}" for i in range(n_resaxes)]) + " ..."
            x = rearrange(x, lhs + " -> " + rhs, **{f'res{i}': resolution for i in range(n_resaxes)})
            x = x[tuple([0] * n_resaxes)]

        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None):
        return *return_value, {"rate": resolution}


    collate_args = ['resolution']

class ImageResolutionCollateMixin(SequenceResolutionCollateMixin):
    """self.collate_fn(resolution, img_size) produces a collate function that resizes inputs to size img_size/resolution"""

    _interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    _antialias = True

    @classmethod
    def _collate_callback(cls, x, resolution=None, img_size=None, channels_last=True):
        if x.ndim < 4:
            return super()._collate_callback(x, resolution=resolution)
        if img_size is None:
            x = super()._collate_callback(x, resolution=resolution)
        else:
            x = rearrange(x, 'b ... c -> b c ...') if channels_last else x
            _size = round(img_size/resolution)
            x = torchvision.transforms.functional.resize(
                x,
                size=[_size, _size],
                interpolation=cls._interpolation,
                antialias=cls._antialias,
            )
            x = rearrange(x, 'b c ... -> b ... c') if channels_last else x
        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None, img_size=None, channels_last=True):
        return *return_value, {"rate": resolution}

    collate_args = ['resolution', 'img_size', 'channels_last']



# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why. So we just provide our own class with the same core methods as LightningDataModule (e.g. setup)
class SequenceDataset(DefaultCollateMixin):
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO it might be possible to write this as a @dataclass, but it seems tricky to separate from the other features of this class such as the _name_ and d_input/d_output
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        kwargs['shuffle'] = 'sampler' not in kwargs # shuffle cant be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_

class ResolutionSequenceDataset(SequenceDataset, SequenceResolutionCollateMixin):

    def _train_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if train_resolution is None: train_resolution = [1]
        if not is_list(train_resolution): train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now."
        return super()._train_dataloader(dataset, resolution=train_resolution[0], **kwargs)

    def _eval_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if dataset is None: return
        if eval_resolutions is None: eval_resolutions = [1]
        if not is_list(eval_resolutions): eval_resolutions = [eval_resolutions]

        dataloaders = []
        for resolution in eval_resolutions:
            dataloaders.append(super()._eval_dataloader(dataset, resolution=resolution, **kwargs))

        return (
            {
                None if res == 1 else str(res): dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None else None
        )

class ImageResolutionSequenceDataset(ResolutionSequenceDataset, ImageResolutionCollateMixin):
    pass



# Registry for dataloader class
loader_registry = {
    None: torch.utils.data.DataLoader, # default case
}
