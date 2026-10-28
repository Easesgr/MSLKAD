"""Single-process CPU / CUDA / DataParallel training and paired evaluation."""

import itertools
import logging
from pathlib import Path
import time

import torch
from torch import nn

from loss.loss import EdgeLoss, PerceptualLoss
from util.checkpoint import resolve_checkpoint, restore_checkpoint
from util.image_io import save_tensor
from util.inference import forward_image, y_metrics
from util.optimizer_scheduler import initialize_optimizer, initialize_scheduler

logger = logging.getLogger(__name__)


def normalize_gpu_ids(value):
    if isinstance(value, str):
        try:
            value = [int(item.strip()) for item in value.split(',') if item.strip()]
        except ValueError as error:
            raise ValueError('train.gpus must contain integer device IDs, e.g. [0,1].') from error
    if type(value) is int:
        value = [value]
    if not isinstance(value, (list, tuple)) or any(type(i) is not int or i < 0 for i in value):
        raise ValueError('train.gpus must be a list of nonnegative integer device IDs.')
    if len(set(value)) != len(value):
        raise ValueError('train.gpus must not contain duplicate device IDs.')
    return list(value)


def training_device(args):
    gpus = normalize_gpu_ids(args.train.gpus)
    if args.sys.cpu or not gpus or not torch.cuda.is_available():
        return torch.device('cpu'), []
    if max(gpus) >= torch.cuda.device_count():
        raise ValueError(f'train.gpus={gpus} exceeds the {torch.cuda.device_count()} visible CUDA devices.')
    return torch.device(f'cuda:{gpus[0]}'), gpus


class Trainer:
    def __init__(self, args, models, dataloaders, ckp):
        self.args, self.ckp = args, ckp
        self.train_dataloader, self.test_dataloader = dataloaders
        self.max_evaluation_count = args.data.max_evaluation_count
        self.device, self.selected_gpus = training_device(args)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.models = {}
        for name, model in models.items():
            model = model.to(self.device)
            if len(self.selected_gpus) > 1:
                model = nn.DataParallel(model, device_ids=self.selected_gpus)
            self.models[name] = model
        logger.info('Using %s, DataParallel devices=%s', self.device, self.selected_gpus)
        self.optimizers, self.schedulers = {}, {}
        self.existing_step, self.current_step, self.current_loss = 0, 0, []
        self.best_psnr = float('-inf')
        if not args.train.test_only:
            if self.train_dataloader is None or len(self.train_dataloader) == 0:
                raise ValueError('Training requires at least one complete batch.')
            for field in ('max_iter', 'print_loss', 'print_lr', 'save_every', 'eval_every'):
                value = args.train[field]
                if type(value) is not int or value < (1 if field == 'max_iter' else 0):
                    raise ValueError(f'train.{field} must be an integer >= {1 if field == "max_iter" else 0}.')
            self.optimizers = {name: initialize_optimizer(args, model, name) for name, model in self.models.items()}
            self.schedulers = {name: initialize_scheduler(args, opt, name) for name, opt in self.optimizers.items()}
        path = resolve_checkpoint(args)
        if path is not None:
            self.existing_step, self.current_loss, state = restore_checkpoint(
                path, self.models, self.optimizers, self.schedulers,
                resume=args.resume and not args.train.test_only,
            )
            self.current_step = self.existing_step
            self.best_psnr = state.get('best_psnr', float('-inf'))
            logger.info('Loaded %s; completed training iterations=%d', path, self.existing_step)
        if not args.train.test_only:
            self.criterionPL = PerceptualLoss(device=self.device, model_path=args.train.vgg_model_path)
            self.criterionML = nn.L1Loss().to(self.device)
            self.criterionEDGE = EdgeLoss(device=self.device).to(self.device)

    def train(self):
        loader = iter(self.train_dataloader)
        last_print_time = time.monotonic()
        for now_iter in range(self.existing_step + 1, self.args.train.max_iter + 1):
            try:
                inputs, targets = next(loader)
            except StopIteration:
                if self.args.train.eval_every == 0 and self.test_dataloader is not None:
                    self.testall(self.current_step)
                loader = iter(self.train_dataloader)
                inputs, targets = next(loader)
            # Evaluation restores mode as well, but set train immediately before forward.
            self.models['MSLKAD'].train()
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)
            self.optimizers['MSLKAD'].zero_grad(set_to_none=True)
            output = self.models['MSLKAD'](inputs)
            loss_pl = self.criterionPL(output, targets)
            loss_ml = self.criterionML(output, targets)
            loss_edge = self.criterionEDGE(output, targets)
            # Preserve the author's actual formula, including MSE perceptual features.
            loss = loss_ml + 0.7 * loss_pl + 0.15 * loss_edge
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at iteration {now_iter}.')
            loss.backward()
            self.optimizers['MSLKAD'].step()
            self.schedulers['MSLKAD'].step()
            self.current_step = now_iter
            self.current_loss = [float(loss.detach())]
            if self.args.train.print_lr and now_iter % self.args.train.print_lr == 0:
                logger.info('Iteration %d learning rate: %.8f', now_iter, self.optimizers['MSLKAD'].param_groups[0]['lr'])
            if self.args.train.print_loss and now_iter % self.args.train.print_loss == 0:
                logger.info('Iter [%d/%d] PL=%.6f L1=%.6f Edge=%.6f Total=%.6f Time=%.2fs',
                            now_iter, self.args.train.max_iter, loss_pl.item(), loss_ml.item(), loss_edge.item(),
                            loss.item(), time.monotonic() - last_print_time)
                last_print_time = time.monotonic()
            final = now_iter == self.args.train.max_iter
            if self.test_dataloader is not None and (final or (
                    self.args.train.eval_every and now_iter % self.args.train.eval_every == 0)):
                self.testall(now_iter)
            if final or (self.args.train.save_every and now_iter % self.args.train.save_every == 0):
                self.save_ckp(now_iter)
        logger.info('Training stopped after %d completed iterations.', self.current_step)

    def evaluate(self, max_samples=None, save_dir=None):
        if self.test_dataloader is None:
            raise ValueError('Evaluation requires a nonempty paired validation dataset.')
        if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
            raise ValueError('max_evaluation_count must be a positive integer.')
        model = self.models['MSLKAD']
        was_training = model.training
        count, psnr_sum, ssim_sum = 0, 0.0, 0.0
        if save_dir is not None:
            save_dir = Path(save_dir).expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)
            # Protect the dataset in the legacy main.py evaluation path too.
            dataset = self.test_dataloader.dataset
            sources = {path.resolve() for pair in getattr(dataset, 'pairs', []) for path in pair}
            if any((save_dir / f'{pair[0].stem}.png').resolve() in sources for pair in getattr(dataset, 'pairs', [])):
                raise ValueError('Output would overwrite an input/target image; choose another save_dir.')
        samples = self.test_dataloader if max_samples is None else itertools.islice(self.test_dataloader, max_samples)
        model.eval()
        try:
            with torch.inference_mode():
                for inputs, targets, filenames in samples:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    output = forward_image(model, inputs, self.args.train.eval_tile_size, self.args.train.eval_tile_overlap)
                    psnr, ssim = y_metrics(output, targets)
                    psnr_sum += psnr
                    ssim_sum += ssim
                    count += 1
                    if save_dir is not None:
                        save_tensor(output, save_dir / f'{Path(filenames[0]).stem}.png')
                    if self.args.train.test_only:
                        logger.info('%s: PSNR(Y)=%.4f SSIM(Y)=%.6f', filenames[0], psnr, ssim)
        finally:
            model.train(was_training)
        if count == 0:
            raise ValueError('Evaluation dataset is empty.')
        result = psnr_sum / count, ssim_sum / count
        logger.info('Dataset %s (%d images): PSNR(Y)=%.4f SSIM(Y)=%.6f',
                    self.args.data.test_data, count, *result)
        return result

    def test(self, now_iter):
        return self.evaluate(max_samples=self.max_evaluation_count)

    def testall(self, now_iter):
        psnr, ssim = self.evaluate()
        if not self.args.train.test_only and psnr > self.best_psnr:
            self.best_psnr = psnr
            self.save_ckp(now_iter, tag='best')
        return psnr, ssim

    def testonly(self):
        return self.evaluate(save_dir=self.args.train.save_dir)

    def tile_forward(self, model, input_tensor, tile_size=256, tile_overlap=32):
        return forward_image(model, input_tensor, tile_size, tile_overlap)

    def save_ckp(self, step, models=None, *, tag=None):
        if self.ckp is not None:
            self.ckp.save_checkpoint(step, self.current_loss, self.models if models is None else models,
                                     self.optimizers, self.schedulers, tag=tag,
                                     trainer_state={'best_psnr': self.best_psnr})
