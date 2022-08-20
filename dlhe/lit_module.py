#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from rich import inspect
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dlhe.model.unet import UNet
from dlhe.model.backboned_unet import BackbonedUNet
from dlhe.utils import plot_heatmaps

import numpy as np
import torch
from torch import Tensor
from torch import nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

class LitModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Using `save_hyperparameters()`  in __init__ will enable lightning to store all the
        # arguments provided in `add_model_specific_ar gs` within the `self.hparams` attribute.
        # These will also be stored within the model checkpoint.
        self.save_hyperparameters()
        # self.model = UNet(
        #     # n_channels=self.hparams.in_channels,
        #     n_classes=self.hparams.out_channels,
        #     # bilinear=True,
        # )
        self.model = BackbonedUNet(
            backbone_name='resnet101',
            pretrained=True,
            encoder_freeze=False,
            classes=self.hparams.out_channels,
            decoder_filters=(256, 128, 64, 32, 16),
        )
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #     in_channels=3, out_channels=49, init_features=32, pretrained=False)

        # self.model.register_parameter("alpha", torch.nn.Parameter(torch.randn(1)))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--lr_decay", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--p_dropout", type=float, default=0.3)

        # UNet related arguments
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=49)
        parser.add_argument("--bilinear", type=bool, default=True)

        return parent_parser

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
            

        dice_loss = 0
        bce_loss = 0
        focal_loss = 0
        for channel in range(y_hat.shape[1]):
            bce_loss += F.binary_cross_entropy_with_logits(y_hat[:, channel, ...], y[:, channel, ...], reduction='mean')
            focal_loss += torchvision.ops.sigmoid_focal_loss(y_hat[:, channel, ...], y[:, channel, ...], reduction='mean')
            dice_loss += 1 - dice_coeff(y_hat[:, channel, ...], y[:, channel, ...], reduce_batch_first=True)
        
        loss = 1 * focal_loss + 0.1 * bce_loss + 0.01 * dice_loss

        self.log("train/bce_loss", bce_loss, prog_bar=False)
        self.log("train/dice_loss", dice_loss, prog_bar=False)
        self.log("train/focal_loss", loss, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        self.log("loss", loss, prog_bar=False)

        if batch_idx % 50 == 0:
            input_image = x[0].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            gt_mask = y[0].squeeze(0).detach().cpu().numpy()
            pred_mask = y_hat[0].squeeze(0).detach().cpu().numpy()

            output_path_gt = Path(f"{self.hparams.log_dir}/train-batch_{batch_idx}_gt.png")
            plot_heatmaps(input_image, gt_mask, output_path_gt)
            output_path_pred = Path(f"{self.hparams.log_dir}/train-batch_{batch_idx}_pred.png")
            plot_heatmaps(input_image, pred_mask, output_path_pred)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])
            self.log("lr", sch.optimizer.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_hat = torch.sigmoid(y_hat)
        
        kps_gt = (y == y.amax((2,3), keepdim=True)).nonzero()[..., 2:]
        kps_pred = (y_hat == y_hat.amax((2,3), keepdim=True)).nonzero()[..., 2:]

        if kps_gt.shape != kps_pred.shape:
            nmrse = np.linalg.norm(x.shape[-2:]) / (x.shape[-1] * x.shape[2])
        else:
            nmrse = np.linalg.norm(kps_gt - kps_pred, axis=1).mean() / (x.shape[-1] * x.shape[2])
        
        self.log("val/nmrse", nmrse, prog_bar=True)
        if batch_idx == 0:
            input_image = x[batch_idx].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gt_mask = y[batch_idx].squeeze(0).cpu().numpy()
            pred_mask = y_hat[batch_idx].squeeze(0).cpu().numpy()

            output_path_gt = Path(f"{self.hparams.log_dir}/val-batch_{batch_idx}_gt.png")
            plot_heatmaps(input_image, gt_mask, output_path_gt)
            output_path_pred = Path(f"{self.hparams.log_dir}/val-batch_{batch_idx}_pred.png")
            plot_heatmaps(input_image, pred_mask, output_path_pred)

        return  nmrse

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, test_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            # lr_decay=self.hparams.lr_decay,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True)
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}
        ]


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description="dataloader")

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    args = parser.parse_args()
    inspect(args)

    model = LitModel(**vars(args))
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
