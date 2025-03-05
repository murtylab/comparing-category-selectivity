"""
Sources:
- https://github.com/sacadena/ptrnets/blob/master/ptrnets/utils/config.py
- https://github.com/sacadena/ptrnets/blob/master/ptrnets/utils/gdrive.py
- https://github.com/sacadena/ptrnets/blob/master/ptrnets/utils/modules.py

Download ptrnets config.toml into models folder and rename ptrnets_config.toml
"""

import errno
import os
import re
import sys
import zipfile
import io
from contextlib import contextmanager
from importlib.resources import Package
from importlib.resources import path
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union
from typing import Sequence

import gdown
import requests
import tomli
import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from gdown.parse_url import parse_url

from compcats.models import utils

PathLike = Union[str, os.PathLike]


class TomlConfig:
    def __init__(self, file_path: PathLike) -> None:
        self._file_path = Path(file_path)
        with self._get_stream() as stream:
            self._dict = tomli.load(stream)

    @contextmanager
    def _get_stream(self) -> Iterator[BinaryIO]:
        if self._file_path.is_file():
            with open(self._file_path, mode="rb") as stream:
                yield io.BytesIO(stream.read())
        else:
            raise ValueError(f"Could not find configuration file {self._file_path}")

    def get_dict(self, key: str) -> Dict[str, Any]:
        if key not in self._dict:
            raise ValueError(f"{key} is not among dict keys")
        d = self._dict.get(key, {})
        if not isinstance(d, dict):
            raise ValueError(f"Not a dictionary: {key}")
        return d

    @property
    def available_keys(self) -> List[str]:
        return list(self._dict.keys())


def get_config_file(package: Package) -> Path:
    with path(package, "ptrnets_config.toml") as f:
        return f


def load_state_dict_from_model_name(
    model_name: str,
    progress: bool = True,
    map_location: Optional[str] = None,
    check_hash: bool = False,
) -> Dict[str, Any]:
    config = TomlConfig(file_path=get_config_file(package=utils))
    urls = config.get_dict("model_weights").get("url", [])
    gdrive_ids = config.get_dict("model_weights").get("gdrive", [])
    map_location = map_location or "cpu"
    if model_name in urls:
        return load_state_dict_from_url(
            urls.get(model_name),
            progress=progress,
            map_location=map_location,
            check_hash=check_hash,
        )
    if model_name in gdrive_ids:
        gid = gdrive_ids.get(model_name)
        return load_state_dict_from_google_drive(
            gid,
            progress=True,
            filename=f"{gid}.pth.tar",
            map_location=map_location,
        )
    raise ValueError(f"Model {model_name} not found in weight config")


def _get_name(id_: str) -> str:
    """
    Gets the base url name from the gdrive using the file id.
    """

    url = f"https://drive.google.com/uc?id={id_}"
    sess = requests.session()
    file_id, is_download_link = parse_url(url)

    while True:
        try:
            res = sess.get(url, stream=True)
        except requests.exceptions.ProxyError as e:
            print("An error has occurred using proxy:", file=sys.stderr)
            print(e, file=sys.stderr)
            return ""

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (file_id and is_download_link):
            break

        url = res.url  # Update the URL if redirected

    if file_id and is_download_link:
        m = re.search('filename="(.*)"', res.headers["Content-Disposition"])
        output = m.groups()[0] if m else ""
    else:
        output = os.path.basename(url)

    return output


def load_state_dict_from_google_drive(
    id_: str,
    model_dir: Optional[str] = None,
    map_location: Optional[str] = None,
    progress: bool = True,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given google drive file id.
    Note: Inspired by torch.hub.load_state_dict_from_url

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if not set.

    Args:
        id_ (string): GDrive file id
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        filename (string, optional): name of the file to store

    Example:
        >>> state_dict = load_state_dict_from_google_drive(id_='18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG')

    """

    if model_dir is None:
        torch_home = torch.hub._get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    if filename is None:
        filename = _get_name(id_)  # use default name of the file in gdrive

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = f"https://drive.google.com/uc?id={id_}"
        gdown.download(url, cached_file, quiet=not (progress))

    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()

            if len(members) != 1:
                raise RuntimeError("Only one file(not dir) is allowed in the zipfile")

            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)

    print(cached_file)
    map_location = map_location or "cpu"
    return torch.load(cached_file, map_location=map_location)

class Unnormalize(nn.Module):
    """
    Helper class for unnormalizing input tensor
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.0,),
        std: Sequence[float] = (1.0,),
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unnormalize(x, self.mean, self.std, self.inplace)


def unnormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.0,),
    std: Sequence[float] = (1.0,),
    inplace: bool = False,
) -> torch.Tensor:
    """Unnormalize a tensor image by first multiplying by std (channel-wise) and then adding the mean (channel-wise)

    Args:
        tensor (Tensor): Tensor image of size (N, C, H, W) to be de-standarized.
        mean (sequence): Sequence of original means for each channel.
        std (sequence): Sequence of original standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalized Tensor image.

    """

    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor should be a torch tensor. Got {type(tensor)}.")

    if tensor.ndimension() != 4:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (N, C, H, W). Got tensor.size() = {tensor.size()}."
        )
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean_tensor = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std_tensor = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if not std_tensor.all():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    if mean_tensor.ndim == 1:
        mean_tensor = mean_tensor[None, :, None, None]
    if std_tensor.ndim == 1:
        std_tensor = std_tensor[None, :, None, None]

    tensor.mul_(std_tensor).add_(mean_tensor)
    return tensor