"""Strict, device-independent checkpoint loading shared by train and inference."""

from collections.abc import Mapping
from pathlib import Path

import torch


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def read_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return torch.load(path, map_location='cpu', weights_only=True)


def extract_state_dict(checkpoint, model_name='MSLKAD'):
    """Accept repository checkpoints, state_dict wrappers, or a raw state dict."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must be a state dictionary or checkpoint mapping.')
    if 'models' in checkpoint:
        models = checkpoint['models']
        if not isinstance(models, Mapping) or model_name not in models:
            raise ValueError(f'Checkpoint does not contain models[{model_name!r}].')
        state = models[model_name]
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Model state dictionary is empty or invalid.')
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Model state dictionary must map parameter names to tensors.')
    # A DataParallel checkpoint prefixes every key. Mixed prefixes are malformed;
    # never silently discard parameters or guess at an incompatible architecture.
    prefixed = [key.startswith('module.') for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('Checkpoint contains mixed DataParallel module. prefixes.')
    strip_prefix = all(prefixed)
    return {key[7:] if strip_prefix else key: value for key, value in state.items()}


def load_model_state(model, checkpoint, model_name='MSLKAD'):
    unwrap_model(model).load_state_dict(extract_state_dict(checkpoint, model_name), strict=True)


def load_weights(model, path):
    load_model_state(model, read_checkpoint(path))


def resolve_checkpoint(args):
    weights, run = args.get('weights'), args.get('load')
    if weights and run:
        raise ValueError('Specify either weights=PATH or load=RUN_NAME, not both.')
    if weights:
        return Path(weights).expanduser()
    if run:
        if not isinstance(run, str):
            raise ValueError('load must name one run; use weights=PATH for a checkpoint file.')
        return Path(args['home_dir']).expanduser() / 'runs' / run / f"{args['load_model']}.pt"
    if args.get('resume') or args['train']['test_only']:
        raise ValueError('Resume/evaluation requires weights=PATH or load=RUN_NAME.')
    return None


def restore_checkpoint(path, models, optimizers=None, schedulers=None, resume=False):
    checkpoint = read_checkpoint(path)
    if resume:
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Resume requires a complete training checkpoint.')
        step = checkpoint.get('step')
        if type(step) is not int or step < 0:
            raise ValueError('Resume requires a nonnegative integer checkpoint step.')
        for key, objects in [('optimizers', optimizers), ('schedulers', schedulers)]:
            states = checkpoint.get(key)
            if not objects or not isinstance(states, Mapping) or set(objects) - set(states):
                raise ValueError(f'Resume requires matching {key} states in the checkpoint.')
        for key, objects in [('optimizer_types', optimizers), ('scheduler_types', schedulers)]:
            if key in checkpoint:
                for name, obj in objects.items():
                    if checkpoint[key].get(name) != type(obj).__name__:
                        raise ValueError(f'Resume {key} mismatch for {name}; use the original training configuration.')
        # Old repository checkpoints have no class metadata. PyTorch scheduler
        # load_state_dict accepts arbitrary fields, so verify the identifying
        # fields before allowing an old schedule to be silently changed.
        schedule_fields = {
            'CosineAnnealingLR': {'T_max', 'eta_min'},
            'MultiStepLR': {'milestones', 'gamma'},
            'StepLR': {'step_size', 'gamma'},
            'ExponentialLR': {'gamma'},
        }
        for name, scheduler in schedulers.items():
            state = checkpoint['schedulers'][name]
            expected = schedule_fields[type(scheduler).__name__]
            identifying = {'T_max', 'eta_min', 'milestones', 'step_size', 'gamma'} & state.keys()
            if identifying != expected:
                raise ValueError(f'Resume scheduler mismatch for {name}; use the original training configuration.')
    for name, model in models.items():
        load_model_state(model, checkpoint, name)
    if resume:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint['optimizers'][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint['schedulers'][name])
        return checkpoint['step'], checkpoint.get('loss', []), checkpoint.get('trainer_state', {})
    return 0, [], {}
