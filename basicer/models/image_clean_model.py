import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from .base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
import math
from basicsr.archs import define_network
loss_module = importlib.import_module('basicsr.losses')
metric_module = importlib.import_module('basicsr.metrics')
import importlib
from os import path as osp
import torch.nn as nn
from basicsr.utils import scandir
import os
import random
import numpy as np
import cv2
import torch.nn.functional as F


from functools import partial
class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)


class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)



class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        if 'train' in self.opt and 'mixing_augs' in self.opt['train']:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        else:
            self.mixing_flag = False
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        print(load_path)
        # exit()
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
            self.cri_amp = AMPLoss().to(self.device)
            self.cri_phase = PhaLoss().to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        # print(self.lq.shape,'train shape===============')
        # exit()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        # print(preds)
        # print(self.net_g)
        # exit()
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_amp = 0.05 * self.cri_amp(pred, self.gt)
            l_pha = 0.05 * self.cri_phase(pred, self.gt)
            l_p = self.cri_pix(pred, self.gt)
            #l_pix += l_p + l_pha
            l_pix += l_p + l_amp + l_pha

        loss_dict['l_p'] = l_p
        loss_dict['l_pha'] = l_pha
        loss_dict['l_amp'] = l_amp

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        print(self.lq.shape)
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        print(img.shape,'098')
        # print(img)
        # exit()
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr=None, use_image=None):
        if rgb2bgr is None:
            rgb2bgr = self.opt['val'].get('rgb2bgr', False)
        if use_image is None:
            use_image = self.opt['val'].get('use_image', False)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.opt['val']['metrics'].keys()}
            self.metric_counts = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        else:
            self.metric_results = {}
            self.metric_counts = {}

        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        num_samples_processed_for_any_metric = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            try:
                test()
            except Exception as e:
                logger = get_root_logger()
                logger.error(f"Error during test() for {img_name}: {e}")
                if hasattr(self, 'lq') and self.lq is not None: del self.lq
                if hasattr(self, 'output') and self.output is not None: del self.output
                if hasattr(self, 'gt') and self.gt is not None: del self.gt
                torch.cuda.empty_cache()
                continue

            visuals = self.get_current_visuals()

            sr_img_numpy = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img_numpy = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
            else:
                logger = get_root_logger()
                logger.warning(f"No GT image for {img_name}, skipping metrics calculation for this sample.")
                if hasattr(self, 'lq') and self.lq is not None: del self.lq
                if hasattr(self, 'output') and self.output is not None: del self.output
                torch.cuda.empty_cache()
                continue

            if hasattr(self, 'lq') and self.lq is not None: del self.lq
            if hasattr(self, 'output') and self.output is not None: del self.output
            if hasattr(self, 'gt') and self.gt is not None: del self.gt
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                imwrite(sr_img_numpy, save_img_path)
                if 'gt' in visuals:
                    imwrite(gt_img_numpy, save_gt_img_path)

            sample_had_successful_metric = False
            if with_metrics:
                opt_metric_config = deepcopy(self.opt['val']['metrics'])
                for metric_name, metric_opt_config in opt_metric_config.items():
                    metric_type = metric_opt_config.pop('type')

                    if use_image:
                        input1 = sr_img_numpy
                        input2 = gt_img_numpy
                    else:
                        input1 = visuals['result']
                        input2 = visuals['gt']

                    try:
                        temp_metric_val = getattr(
                            metric_module, metric_type)(input1, input2, **metric_opt_config)

                        if not math.isnan(temp_metric_val) and temp_metric_val != float('inf'):
                            self.metric_results[metric_name] += temp_metric_val
                            self.metric_counts[metric_name] += 1
                            sample_had_successful_metric = True
                            # print(f"Sample {idx+1} ({img_name}), Metric {metric_name}: current_val={temp_metric_val:.4f}, "
                            #       f"accumulated={self.metric_results[metric_name]:.4f}, "
                            #       f"count_for_this_metric={self.metric_counts[metric_name]}")
                        else:
                            logger = get_root_logger()
                            logger.warning(
                                f"Metric {metric_name} for {img_name} is NaN or Inf. Value: {temp_metric_val}. Skipping.")
                    except Exception as e:
                        logger = get_root_logger()
                        logger.error(f"Error calculating metric {metric_name} for {img_name}: {e}")

            if sample_had_successful_metric:
                num_samples_processed_for_any_metric += 1

        returned_main_metric_value = 0.0  # 修正变量名

        if with_metrics:
            averaged_metrics = {}
            for metric_name in self.metric_results.keys():
                if self.metric_counts[metric_name] > 0:
                    averaged_metrics[metric_name] = self.metric_results[metric_name] / self.metric_counts[metric_name]
                else:
                    averaged_metrics[metric_name] = 0.0

            self.metric_results = averaged_metrics

            counts_log_str = "Metric counts: "
            for metric_name, count_val in self.metric_counts.items():
                counts_log_str += f"{metric_name}={count_val}, "
            logger = get_root_logger()
            logger.info(counts_log_str)
            logger.info(
                f"Total samples in dataloader: {len(dataloader)}. Samples processed for at least one metric: {num_samples_processed_for_any_metric}")

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            main_metric_name = self.opt['val'].get('main_metric', 'psnr')
            returned_main_metric_value = self.metric_results.get(main_metric_name, 0.0)  # 修正变量名

        return returned_main_metric_value  # 修正变量名


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            log_str += f' (on {self.metric_counts.get(metric, 0)} samples)'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                if self.metric_counts.get(metric, 0) > 0:
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        # print(self.lq)
        # print(self.output)
        # exit()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
