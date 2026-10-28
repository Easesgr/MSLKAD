"""Run logging and complete, portable training checkpoints."""

import logging
from datetime import datetime
from pathlib import Path
import sys

import torch
import yaml

from util.checkpoint import resolve_checkpoint, restore_checkpoint, unwrap_model

logger = logging.getLogger(__name__)


class Checkpoint:
    def __init__(self, args):
        if args.get('reset'):
            raise ValueError('reset is not supported; choose a new run name instead.')
        self.args = args
        self.name = f"{args['name']}-{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
        self.log_dir = str(Path(args['home_dir']).expanduser() / 'runs' / self.name)
        Path(self.log_dir).mkdir(parents=True, exist_ok=False)
        logging.basicConfig(
            level=logging.DEBUG if args['debug'] else logging.INFO,
            handlers=[logging.FileHandler(Path(self.log_dir) / 'run.log'), logging.StreamHandler(sys.stdout)],
        )
        with (Path(self.log_dir) / 'configs.yaml').open('w', encoding='utf-8') as file:
            yaml.safe_dump(args, file, sort_keys=False)
        logger.info('Checkpoint directory: %s', self.log_dir)

    def load_checkpoint(self, models, optimizers=None, schedulers=None):
        path = resolve_checkpoint(self.args)
        if path is None:
            return 0, []
        step, loss, _ = restore_checkpoint(path, models, optimizers, schedulers, self.args['resume'])
        return step, loss

    def save_checkpoint(self, step, loss, models, optimizers, schedulers=None, *, tag=None, trainer_state=None):
        if type(step) is not int or step < 0:
            raise ValueError('Checkpoint step must be a nonnegative integer, including for best checkpoints.')
        name = str(step) if tag is None else tag
        if name not in (str(step), 'best'):
            raise ValueError('Checkpoint tag must be best or omitted.')
        checkpoint = {
            'step': step,
            'loss': loss,
            'models': {name: unwrap_model(model).state_dict() for name, model in models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
            'schedulers': {name: scheduler.state_dict() for name, scheduler in (schedulers or {}).items()},
            'optimizer_types': {name: type(opt).__name__ for name, opt in optimizers.items()},
            'scheduler_types': {name: type(scheduler).__name__ for name, scheduler in (schedulers or {}).items()},
            'trainer_state': trainer_state or {},
        }
        for filename in (f'{name}.pt', 'latest.pt'):
            path = Path(self.log_dir) / filename
            temporary = path.with_suffix('.pt.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        self.write_step(step)
        logger.info('Saved %s.pt and latest.pt at iteration %d', name, step)

    def write_step(self, step):
        (Path(self.log_dir) / 'step.txt').write_text(str(step), encoding='utf-8')
