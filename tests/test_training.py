import copy

import pytest
import torch
from torch.utils.data import DataLoader

from log import Checkpoint
from loss.loss import EdgeLoss, PerceptualLoss
from model import get_models
from trainer.train import Trainer, normalize_gpu_ids, training_device
from util.checkpoint import read_checkpoint, restore_checkpoint
from util.optimizer_scheduler import initialize_optimizer, initialize_scheduler


def test_perceptual_gradients_reach_generator_but_not_vgg_or_target(vgg_weights):
    criterion = PerceptualLoss(device='cpu', model_path=vgg_weights)
    prediction = torch.rand(1, 3, 8, 9, requires_grad=True)
    target = torch.rand(1, 3, 8, 9, requires_grad=True)
    loss = criterion(prediction, target)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in criterion.parameters())
    with torch.no_grad():
        expected = sum(torch.nn.functional.mse_loss(a, b) for a, b in zip(
            criterion.get_layer_output(prediction), criterion.get_layer_output(target))) / 5
    torch.testing.assert_close(loss, expected)


def test_vgg_is_required_and_edge_kernel_follows_module_device_dtype(tmp_path):
    with pytest.raises(FileNotFoundError, match='VGG19'):
        PerceptualLoss(model_path=tmp_path / 'missing.pth')
    edge = EdgeLoss(device='cpu').to(dtype=torch.float64)
    assert 'kernel' in dict(edge.named_buffers())
    assert edge.kernel.dtype == torch.float64
    prediction = torch.rand(1, 3, 7, 9, dtype=torch.float64, requires_grad=True)
    edge(prediction, torch.zeros_like(prediction)).backward()
    assert prediction.grad.abs().sum() > 0


def test_cpu_override_and_legacy_gpu_ids(args, monkeypatch):
    assert normalize_gpu_ids('0,1,2,3') == [0, 1, 2, 3]
    assert normalize_gpu_ids((1, 0)) == [1, 0]
    for value in [[0, 0], [-1], 'cpu', [False]]:
        with pytest.raises(ValueError):
            normalize_gpu_ids(value)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert training_device(args) == (torch.device('cpu'), [])
    args.sys.cpu = False
    args.train.gpus = [1, 0]
    assert training_device(args) == (torch.device('cuda:1'), [1, 0])
    args.train.gpus = [2]
    with pytest.raises(ValueError, match='visible CUDA'):
        training_device(args)


def test_training_resume_matches_uninterrupted_steps(args, vgg_weights):
    args.train.vgg_model_path = vgg_weights
    args.model.MSLKAD.scheduler.T_max = 4
    torch.manual_seed(42)
    initial_models = get_models(args)
    inputs = torch.rand(3, 8, 9)
    targets = torch.rand(3, 8, 9)
    loaders = (DataLoader([(inputs, targets)], batch_size=1),
               DataLoader([(inputs, targets, 'sample.png')], batch_size=1))
    uninterrupted = Trainer(copy.deepcopy(args), copy.deepcopy(initial_models), loaders, None)
    uninterrupted.train()
    first_args = copy.deepcopy(args)
    first_args.train.max_iter = 2
    # Checkpoint accepts the plain dictionary emitted by the config parser.
    # Convert recursive AttrDict values without serializing Python-specific YAML tags.
    def plain(value):
        return {key: plain(item) for key, item in value.items()} if isinstance(value, dict) else value
    ckp = Checkpoint(plain(first_args))
    first = Trainer(first_args, copy.deepcopy(initial_models), loaders, ckp)
    first.train()
    checkpoint = read_checkpoint(f'{ckp.log_dir}/latest.pt')
    assert checkpoint['step'] == 2
    assert checkpoint['schedulers']['MSLKAD']['last_epoch'] == 2
    assert all(not key.startswith('module.') for key in checkpoint['models']['MSLKAD'])
    assert read_checkpoint(f'{ckp.log_dir}/best.pt')['step'] in (1, 2)
    resume_args = copy.deepcopy(args)
    resume_args.weights = f'{ckp.log_dir}/latest.pt'
    resume_args.resume = True
    resumed = Trainer(resume_args, get_models(resume_args), loaders, None)
    assert resumed.existing_step == 2
    assert resumed.best_psnr == first.best_psnr
    resumed.train()
    assert resumed.current_step == uninterrupted.current_step == 4
    assert resumed.schedulers['MSLKAD'].state_dict() == uninterrupted.schedulers['MSLKAD'].state_dict()
    assert resumed.optimizers['MSLKAD'].param_groups[0]['lr'] == pytest.approx(1e-6)
    for name, value in uninterrupted.models['MSLKAD'].state_dict().items():
        torch.testing.assert_close(resumed.models['MSLKAD'].state_dict()[name], value, rtol=0, atol=0)
    # Evaluation at data-loader boundaries must not leave the model in eval mode.
    assert resumed.models['MSLKAD'].training


def test_testonly_needs_neither_training_data_nor_vgg_and_counts_actual_samples(args, tmp_path, monkeypatch):
    args.train.test_only = True
    args.train.vgg_model_path = str(tmp_path / 'missing-vgg.pth')
    args.data.train_data_root = str(tmp_path / 'missing-training-data')
    args.train.save_dir = str(tmp_path / 'results')
    models = get_models(args)
    weights = tmp_path / 'model.pt'
    torch.save(models['MSLKAD'].state_dict(), weights)
    args.weights = str(weights)
    image = torch.rand(1, 3, 8, 9)
    with torch.no_grad():
        target = models['MSLKAD'](image)[0]
    loader = DataLoader([(image[0], target, 'same.png')], batch_size=1)
    trainer = Trainer(args, models, (None, loader), None)
    assert trainer.optimizers == {}
    assert not hasattr(trainer, 'criterionPL')
    trainer.models['MSLKAD'].train()
    psnr, ssim = trainer.test(0)  # max_evaluation_count=10, but only one image exists.
    assert psnr == float('inf')
    assert ssim == pytest.approx(1.0)
    assert trainer.models['MSLKAD'].training
    trainer.testonly()
    assert (tmp_path / 'results' / 'same.png').is_file()


def test_legacy_scheduler_mismatch_and_missing_optimizer_fail(args, tmp_path):
    models = get_models(args)
    optimizer = initialize_optimizer(args, models['MSLKAD'], 'MSLKAD')
    cosine = initialize_scheduler(args, optimizer, 'MSLKAD')
    legacy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.5)
    payload = {
        'step': 1, 'models': {'MSLKAD': models['MSLKAD'].state_dict()},
        'optimizers': {'MSLKAD': optimizer.state_dict()}, 'schedulers': {'MSLKAD': legacy.state_dict()},
    }
    path = tmp_path / 'legacy.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='scheduler mismatch'):
        restore_checkpoint(path, models, {'MSLKAD': optimizer}, {'MSLKAD': cosine}, resume=True)
    # The historical scheduler still resumes when configured explicitly.
    step, _, _ = restore_checkpoint(path, models, {'MSLKAD': optimizer}, {'MSLKAD': legacy}, resume=True)
    assert step == 1
    del payload['optimizers']
    torch.save(payload, path)
    with pytest.raises(ValueError, match='optimizers'):
        restore_checkpoint(path, models, {'MSLKAD': optimizer}, {'MSLKAD': legacy}, resume=True)
