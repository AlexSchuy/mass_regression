import os

import pytorch_lightning as pl


class CheckpointEveryNSteps(pl.Callback):

    def __init__(self, save_step_frequency, prefix="N-Step-Checkpoint", use_modelcheckpoint_filename=False):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f'{self.prefix}_{epoch}_{global_step}.ckpt'
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
