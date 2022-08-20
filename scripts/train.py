import os
from argparse import ArgumentParser
from gc import callbacks
from pathlib import Path
from time import gmtime, strftime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f"./log/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}",
    )
    # parser.add_argument("--batch_size", type=int, default=8)
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

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=Path(args.log_dir) / "checkpoints")
    callbacks = [checkpoint_callback]

    # see training flags for automatically added args
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    trainer = pl.Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)

    # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, datamodule=dm)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig(Path(args.log_dir) / "lr_finder.png")

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(f"New learning rate: {new_lr}")

    # # update hparams of the model
    # model.hparams.lr = new_lr

    # Start training
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(model, dm.test_dataloader())

    print("finished.")
