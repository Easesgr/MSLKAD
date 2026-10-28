from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from model import get_models
from util.checkpoint import extract_state_dict, load_weights, resolve_checkpoint, restore_checkpoint
from util.inference import forward_image
from util.sys_util import read_yaml


def test_default_architecture_count_and_odd_image_shape():
    config = read_yaml(Path(__file__).resolve().parents[1] / 'configs/default.yaml')
    model = get_models(config)['MSLKAD'].eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_295_923
    assert len(model.state_dict()) == 1486
    assert model.body[0].body_t.att[1].dilation == (3, 3)
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 7, 9))
    assert output.shape == (1, 3, 7, 9)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('format', ['raw', 'repository', 'state_dict', 'model_state_dict'])
@pytest.mark.parametrize('parallel', [False, True])
def test_strict_roundtrip(args, tmp_path, format, parallel):
    torch.manual_seed(4)
    source = get_models(args)['MSLKAD'].eval()
    # Nonzero residual scales exercise attention branches, not just initial shortcuts.
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith('scale'):
                parameter.fill_(0.2)
    state = source.state_dict()
    if parallel:
        state = OrderedDict((f'module.{name}', value) for name, value in state.items())
    payload = {'models': {'MSLKAD': state}} if format == 'repository' else state if format == 'raw' else {format: state}
    path = tmp_path / 'weights.pt'
    torch.save(payload, path)
    destination = get_models(args)['MSLKAD'].eval()
    load_weights(destination, path)
    inputs = torch.rand(1, 3, 13, 17)
    with torch.inference_mode():
        torch.testing.assert_close(source(inputs), destination(inputs), rtol=0, atol=0)
    # Loading into a wrapped model must work with either checkpoint spelling too.
    wrapped = torch.nn.DataParallel(get_models(args)['MSLKAD'])
    load_weights(wrapped, path)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(wrapped.module.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'shape', 'mixed_prefix', 'wrong_model'])
def test_incompatible_checkpoints_fail(args, tmp_path, damage):
    model = get_models(args)['MSLKAD']
    state = dict(model.state_dict())
    if damage == 'missing':
        del state['head.weight']
    elif damage == 'extra':
        state['unexpected.weight'] = torch.ones(1)
    elif damage == 'shape':
        state['head.weight'] = torch.ones(1)
    elif damage == 'mixed_prefix':
        state['module.head.weight'] = state.pop('head.weight')
    else:
        state = {'models': {'wrong': state}}
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_weights(model, path)


def test_missing_weights_and_resume_metadata_fail(args, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_weights(get_models(args)['MSLKAD'], tmp_path / 'absent.pt')
    with pytest.raises(ValueError, match='empty'):
        extract_state_dict({})
    args.resume = True
    with pytest.raises(ValueError, match='requires'):
        resolve_checkpoint(args)
    path = tmp_path / 'weights.pt'
    models = get_models(args)
    torch.save(models['MSLKAD'].state_dict(), path)
    with pytest.raises(ValueError, match='step'):
        restore_checkpoint(path, models, {}, {}, resume=True)
    torch.save({'step': 'all_best_4', 'models': {'MSLKAD': models['MSLKAD'].state_dict()}}, path)
    with pytest.raises(ValueError, match='integer'):
        restore_checkpoint(path, models, {}, {}, resume=True)


def test_tiling_covers_small_and_odd_images_without_padding():
    model = torch.nn.Identity()
    for shape in [(1, 3, 1, 3), (1, 3, 13, 19), (1, 3, 8, 8)]:
        inputs = torch.rand(shape)
        torch.testing.assert_close(forward_image(model, inputs, 8, 3), inputs)
    with pytest.raises(ValueError, match='tile_overlap'):
        forward_image(model, inputs, 8, 8)
