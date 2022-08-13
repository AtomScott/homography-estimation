import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from rich import inspect

from dlhe.data_module import KeypointDataModule
from dlhe.lit_module import LitModel

# ----------------
# trainer_main.py
# ----------------

if __name__ == "__main__":
    pl.seed_everything(42069)

    # ------------
    # args
    # ------------
    # add PROGRAM level args
    # (args for the data module are included here)
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--shuffle", type=bool, default=True)

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # parse the args & print them
    args = parser.parse_args()
    kwargs = vars(args)
    inspect(args, docs=False, value=False)

    # -----------------
    # setup data module
    # -----------------

    dm = KeypointDataModule(args.data_dir, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    model = LitModel(**kwargs)

    # ------------
    # training
    # ------------
    # setup loggers
    name = ""
    csv_logger = CSVLogger(args.log_dir, name=name, version="")
    tb_logger = TensorBoardLogger(args.log_dir, name=name, version="")
    loggers = [csv_logger, tb_logger]
    os.makedirs(args.log_dir, exist_ok=True)

    # see training flags for automatically added args
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    trainer = pl.Trainer.from_argparse_args(args, logger=loggers)

    # Start training
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(model, dm.test_dataloader())

    print("finished.")
