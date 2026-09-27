import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_250epochs_preview(nnUNetTrainer_250epochs):
    """250-epoch trainer that also dumps input/target/prediction PNGs every N epochs.

    NNUNET_PREVIEW_EVERY sets the interval (default 10). NNUNET_PREVIEW_DIR sets where
    previews are written (default <output_folder>/previews). Checkpointing is aligned to
    the same interval so checkpoint_latest.pth tracks the previews.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.preview_every = int(os.environ.get('NNUNET_PREVIEW_EVERY', 10))
        self.preview_dir = os.environ.get('NNUNET_PREVIEW_DIR')
        self.save_every = min(self.save_every, self.preview_every)

    def on_epoch_end(self):
        completed = self.current_epoch          # super() increments after this
        super().on_epoch_end()
        if self.local_rank == 0 and (completed + 1) % self.preview_every == 0:
            self._save_preview(completed)

    def _save_preview(self, completed_epoch):
        prev_dir = self.preview_dir if self.preview_dir else join(self.output_folder, 'previews')
        maybe_mkdir_p(prev_dir)

        batch = next(self.dataloader_val)
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, (list, tuple)):
            target = target[0]

        self.network.eval()
        with torch.no_grad():
            ctx = autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context()
            with ctx:
                output = self.network(data)
        if isinstance(output, (list, tuple)):
            output = output[0]

        img  = data[0, 0].detach().cpu().float().numpy()
        gt   = target[0, 0].detach().cpu().numpy()
        pred = output.argmax(1)[0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img, cmap='gray');            ax[0].set_title('input (bf)')
        ax[1].imshow(gt,  vmin=0, vmax=2);         ax[1].set_title('target (0/1/ignore=2)')
        ax[2].imshow(pred, vmin=0, vmax=1);        ax[2].set_title('prediction')
        for a in ax:
            a.axis('off')
        fig.suptitle(f'epoch {completed_epoch + 1}')
        fig.tight_layout()
        out = join(prev_dir, f'preview_epoch_{completed_epoch + 1:04d}.png')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        self.print_to_log_file(f'[preview] saved previews/preview_epoch_{completed_epoch + 1:04d}.png')
