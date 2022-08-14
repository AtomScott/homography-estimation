#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from rich import inspect
from torch.nn import functional as F

from dlhe.model.unet import UNet


class LitModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Using `save_hyperparameters()`  in __init__ will enable lightning to store all the
        # arguments provided in `add_model_specific_ar gs` within the `self.hparams` attribute.
        # These will also be stored within the model checkpoint.
        self.save_hyperparameters()
        self.model = UNet(
            n_channels=self.hparams.in_channels,
            n_classes=self.hparams.out_channels,
            bilinear=True,
        )
        # self.model.register_parameter("alpha", torch.nn.Parameter(torch.randn(1)))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=0.003)
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
        # loss = dice_loss(y_hat, y, multiclass=True)
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train/bce_loss", bce_loss, prog_bar=True)
        return bce_loss

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val/bce_loss", bce_loss, prog_bar=True)
        return bce_loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, test_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(
            self.parameters(),
            lr=self.hparams.lr,
            lr_decay=self.hparams.lr_decay,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


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
