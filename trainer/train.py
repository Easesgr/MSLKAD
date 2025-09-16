import torch
import torch.nn as nn

import logging
import itertools

import torch.nn.functional as F

logger = logging.getLogger(__name__)
from util.optimizer_scheduler import initialize_optimizer, initialize_scheduler
from util.metric_util import  tensor2img, calculate_psnr, calculate_ssim

import matplotlib
matplotlib.use('Agg')


from loss.loss import  PerceptualLoss,EdgeLoss
import time
import os
import torchvision.utils as vutils
from pytorch_msssim import SSIM, MS_SSIM
class Trainer():
    def __init__(self, args, models, dataloaders, ckp):
        self.args = args
        self.ckp = ckp
        self.train_dataloader, self.test_dataloader = dataloaders  # 训练和测试数据加载器
        # 设定测试时最多使用的数据集图片数
        self.max_evaluation_count = self.args.data.max_evaluation_count
        # 指定使用 GPU
        self.selected_gpus = args.train.gpus

        # 判断设备
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.selected_gpus[0]}")
            torch.cuda.set_device(self.selected_gpus[0])
        else:
            self.device = torch.device("cpu")

        # 多 GPU 适配
        self.models = {}
        for name, model in models.items():
            if torch.cuda.device_count() > 1 and torch.cuda.is_available() and len(self.selected_gpus) > 1:
                model = nn.DataParallel(model, device_ids=self.selected_gpus)
            model.to(self.device)
            self.models[name] = model

        # 定义优化器和调度器
        self.optimizers = {name: initialize_optimizer(args, model, name) for name, model in self.models.items()}
        self.schedulers = {name: initialize_scheduler(args, optimizer, name) for name, optimizer in
                           self.optimizers.items()}

        self.existing_step = 0  # 记录已有的训练步数
        self.current_loss = []  # 存储当前的损失值
        # 初始化数据迭代器
        self.loader_train_iter = iter(self.train_dataloader)

        # ----------------------------------
        # 内容感知loss
        self.criterionPL = PerceptualLoss(device=self.device, model_path=self.args.train.vgg_model_path)
        # 鲁棒性L1loss
        self.criterionML = nn.L1Loss().to(self.device)
        # 边缘loss
        self.criterionEDGE = EdgeLoss(device=self.device).to(self.device)
        ### Model Loading
        self.load_previous_ckp(models=self.models)

        # 计算测试轮数和保存模型轮数
        self.test_every = 1
        self.best_psnr = 0

        self.all_best_psnr = 0
        self.best_model_root = os.path.join(self.ckp.log_dir, '0.pth')
        self.all_best_model_root = os.path.join(self.ckp.log_dir, f"all_best_0.pt")

    def load_previous_ckp(self, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.models

            self.existing_step, self.current_loss = self.ckp.load_checkpoint(models, self.optimizers, self.schedulers)

            if self.existing_step > 0:
                logger.info('Resuming training.')

    def train(self):
        now_iter = 1
        last_print_time = time.time()  # ⏱️ 记录上一次 print_loss 的时间
        while now_iter <= self.args.train.max_iter:
            self.models['MSLKAD'].train()
            try:
                lr, hr = next(self.loader_train_iter)
            except StopIteration:
                self.loader_train_iter = iter(self.train_dataloader)
                logger.info(f'Iter {now_iter} Resuming Training Data.')
                # 每三个测全部
                if self.test_every % 1 == 0 :
                    self.testall(now_iter)

                lr, hr = next(self.loader_train_iter)
                self.test_every += 1

            lr_tensor = lr.to(self.device, dtype=torch.float32)
            hr_tensor = hr.to(self.device, dtype=torch.float32)

            out = self.models['MSLKAD'](lr_tensor)

            self.optimizers['MSLKAD'].zero_grad()

            loss_PL = self.criterionPL(out, hr_tensor.detach())

            loss_ML = self.criterionML(out, hr_tensor)
            loss_edge = self.criterionEDGE(out, hr_tensor)
            # loss_fft = self.criterionFFT(out, hr_tensor)
            loss_G = loss_ML +  0.7 *loss_PL + 0.15 * loss_edge

            loss_G.backward()
            self.optimizers['MSLKAD'].step()
            self.schedulers['MSLKAD'].step()

            if now_iter % self.args.train.print_lr == 0:
                current_lr = self.optimizers['MSLKAD'].param_groups[0]['lr']
                logger.info(f"Current LR: {current_lr:.6f}")

            # 每 print_loss 次打印一次损失和耗时（区间耗时）
            if now_iter % self.args.train.print_loss == 0:
                current_time = time.time()
                interval_time = current_time - last_print_time
                interval_str = time.strftime('%H:%M:%S', time.gmtime(interval_time))

                logger.info(
                    'Iter [{:04d}/{}]\t'
                    'Raw Losses [PL: {:.3f} | ML: {:.3f} | Edge: {:.3f} ]  '
                    'Weighted Total: {:.3f}\t'
                    'Time: {}'.format(
                        now_iter, self.args.train.max_iter,
                        loss_PL.item(), loss_ML.item(), loss_edge.item(),
                        loss_G.item(),
                        interval_str
                    )
                )

                last_print_time = current_time  # 🔄 更新为本次时间

            if now_iter == self.args.train.max_iter:
                self.save_ckp(now_iter, self.models)
            now_iter += 1

    def test(self, now_iter):
        self.models['MSLKAD'].eval()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            max_test_samples = self.max_evaluation_count
            start_time = time.time()

            for lr, hr, filename in itertools.islice(self.test_dataloader, max_test_samples):
                lr_tensor = lr.clone().detach().to(dtype=torch.float32, device=self.device)
                height, width = lr_tensor.shape[2], lr_tensor.shape[3]
                out = self.models['MSLKAD'](lr_tensor)

                sr = out[:, :, :height, :width]  # 防止模型输出略大（保险起见保留）

                sr = tensor2img(sr)
                hr = tensor2img(hr)

                now_psnr = calculate_psnr(sr, hr,crop_border=0,input_order="HWC")
                now_ssim = calculate_ssim(sr, hr,crop_border=0,input_order="HWC")

                eval_psnr += now_psnr
                eval_ssim += now_ssim

            end_time = time.time()
            test_time = end_time - start_time

            avg_psnr = eval_psnr / max_test_samples
            avg_ssim = eval_ssim / max_test_samples

            # 更新最优模型逻辑
            if avg_psnr > self.best_psnr:
                logger.info(f"[now_iter {now_iter}] New best model found! PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")

                # 删除旧的最优模型
                if os.path.exists(self.best_model_root):
                    os.remove(self.best_model_root)

                # 保存新的最优模型
                best_model_path = os.path.join(self.ckp.log_dir, f"{now_iter}.pt")
                self.save_ckp(now_iter, models=self.models)

                # 更新状态
                self.best_psnr = avg_psnr

                self.best_model_root = best_model_path
                # 发现最好的测试全部
                #self.testall(now_iter)

            logger.info('[now_iter {}]\tPSNR: {:.3f} SSIM: {:.4f} | Test Time: {:.2f}s'.format(
                self.args.data.test_data,
                avg_psnr,
                avg_ssim,
                test_time
            ))

    def tile_forward(self, model, input_tensor, tile_size=256, tile_overlap=32):
        """
        滑窗推理（无缩放），用于大图分块处理。

        Args:
            model: 网络模型
            input_tensor: 输入图像 tensor，形状 [B, C, H, W]
            tile_size: 每块 tile 的边长
            tile_overlap: 邻近 tile 的重叠区域

        Returns:
            output_tensor: 拼接后的完整图像（与输入尺寸一致）
        """
        B, C, H, W = input_tensor.shape
        stride = tile_size - tile_overlap

        E = torch.zeros_like(input_tensor)  # 拼接输出
        W_map = torch.zeros_like(input_tensor)  # 加权融合计数

        h_idx_list = list(range(0, H - tile_size, stride)) + [H - tile_size] if H > tile_size else [0]
        w_idx_list = list(range(0, W - tile_size, stride)) + [W - tile_size] if W > tile_size else [0]

        for y in h_idx_list:
            for x in w_idx_list:
                in_patch = input_tensor[:, :, y:y + tile_size, x:x + tile_size]
                patch_h, patch_w = in_patch.shape[2:]

                # patch 可能不足 tile_size，需 pad
                pad_bottom = tile_size - patch_h
                pad_right = tile_size - patch_w
                if pad_bottom > 0 or pad_right > 0:
                    in_patch = F.pad(in_patch, (0, pad_right, 0, pad_bottom), mode='reflect')

                with torch.no_grad():
                    out_patch = model(in_patch)
                    out_patch = out_patch[:, :, :patch_h, :patch_w]  # 截断 padding 区域

                E[:, :, y:y + patch_h, x:x + patch_w] += out_patch
                W_map[:, :, y:y + patch_h, x:x + patch_w] += 1

        W_map = torch.where(W_map == 0, torch.ones_like(W_map), W_map)
        return E / W_map
    def testonly(self):
        # 测试
        self.models['MSLKAD'].eval()
        save_dir = self.args.train.save_dir
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for idx, (lr, hr, filename) in enumerate(self.test_dataloader):
                lr_tensor = lr.clone().detach().to(dtype=torch.float32, device=self.device)
                hr_tensor = hr.clone().detach().to(dtype=torch.float32, device=self.device)
                height, width = lr_tensor.shape[2], lr_tensor.shape[3]

                out, feature = self.models['MSLKAD'](lr_tensor, lr_tensor - hr_tensor)
                sr = out[:, :, :height, :width]

                # 如果 filename 是字符串
                if isinstance(filename, (list, tuple)):
                    filename = filename[0]  # 假设 dataloader 返回的是 [name]

                save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.png")

                # 归一化到 [0,1]，保存图片
                sr_clamped = torch.clamp(sr, 0, 1)
                vutils.save_image(sr_clamped, save_path)
                print(f"Saved: {save_path}")
    def testall(self,now_iter):
        # 测试
        self.models['MSLKAD'].eval()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            for idx, (lr, hr, filename) in enumerate(self.test_dataloader):
                lr_tensor = lr.clone().detach().to(dtype=torch.float32, device=self.device)
                height, width = lr_tensor.shape[2], lr_tensor.shape[3]

                if width > 1000 or height > 1000:
                    # 使用滑窗分块推理，不做padding
                    out = self.tile_forward(self.models['MSLKAD'], lr_tensor, tile_size=256, tile_overlap=32)
                else:

                    out = self.models['MSLKAD'](lr_tensor)

                sr = out[:, :, :height, :width]  # 保留裁剪，防止 tile 推理超出范围

                sr_img = tensor2img(sr)
                hr_img = tensor2img(hr)

                now_psnr = calculate_psnr(sr_img, hr_img,crop_border=0,input_order="HWC")
                now_ssim = calculate_ssim(sr_img, hr_img,crop_border=0,input_order="HWC")

                eval_psnr += now_psnr
                eval_ssim += now_ssim

                if self.args.train.test_only:
                    logger.info(f"[Sample {idx} - {filename}] PSNR: {now_psnr:.3f} | SSIM: {now_ssim:.4f}")


            avg_psnr = eval_psnr / len(self.test_dataloader)
            avg_ssim = eval_ssim / len(self.test_dataloader)
            # 更新最优模型逻辑
            if avg_psnr > self.all_best_psnr:
                logger.info(f"[now_iter {now_iter}] New best model found! PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")

                # 删除旧的最优模型
                if os.path.exists(self.all_best_model_root):
                    os.remove(self.all_best_model_root)

                # 保存新的最优模型
                best_model_path = os.path.join(self.ckp.log_dir, f"all_best_{now_iter}.pt")
                self.save_ckp(f"all_best_{now_iter}", models=self.models)

                # 更新状态
                self.all_best_psnr = avg_psnr

                self.all_best_model_root = best_model_path

            logger.info('[Dataset {}]\tAvg PSNR: {:.3f} Avg SSIM: {:.4f}'.format(
                self.args.data.test_data,
                avg_psnr,
                avg_ssim,
            ))


    # 保存检查点
    def save_ckp(self, step, models=None):
        if self.ckp is not None:
            if models is None:
                models = self.model
            self.ckp.save_checkpoint(step, self.current_loss, models, self.optimizers, self.schedulers)

    # 记录当前训练步数
    def log_current_step(self, step):
        if self.ckp is not None:
            self.ckp.write_step(step)

    # 记录中间训练结果
    def dump_intermediate_results(self, img, fn, step):
        if self.ckp is not None:
            self.ckp.dump_intermediate_results(img, f'{fn}_{step}.png')


