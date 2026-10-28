from pathlib import Path

import pytest
import torch
from torch import nn

from util.sys_util import AttrDict, read_yaml


@pytest.fixture(autouse=True, scope='session')
def cpu_threads():
    # Tiny convolutional smoke tests are faster and reproducible with one thread.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def args(tmp_path):
    config = AttrDict(read_yaml(Path(__file__).resolve().parents[1] / 'configs/default.yaml'))
    config.home_dir = str(tmp_path)
    config.sys.cpu = True
    config.sys.log = False
    config.model.MSLKAD.model_args = {'n_resblocks': 1, 'n_resgroups': 2, 'n_feats': 6}
    config.train.max_iter = 4
    config.train.print_loss = 0
    config.train.print_lr = 0
    config.train.save_every = 0
    config.train.eval_every = 0
    config.data.n_threads = 0
    config.data.batch_size = 1
    config.data.patch_size = 16
    return config


@pytest.fixture
def vgg_weights(tmp_path, monkeypatch):
    class SmallVGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*[
                layer for _ in range(6) for layer in (nn.Conv2d(3, 3, 3, padding=1), nn.ReLU(inplace=True))
            ])

    torch.manual_seed(123)
    path = tmp_path / 'test-vgg.pth'
    torch.save(SmallVGG().state_dict(), path)
    monkeypatch.setattr('loss.loss.models.vgg19', lambda **kwargs: SmallVGG())
    return str(path)
