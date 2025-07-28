from copy import deepcopy

import torch
from loguru import logger


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        初始化早停机制。

        Args:
            patience (int): 当验证集上的性能在多少个epoch内没有改善时停止训练。
            verbose (bool): 是否打印详细信息。
            delta (float): 最小变化量，用于判断性能是否改善。
        """
        if patience < 0 or not isinstance(patience, int):
            raise ValueError("Patience must be a non-negative integer.")

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.state_dict = {}

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.inf

    @torch.no_grad()
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.debug(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                logger.info(f"Early stopping, best val_loss: {val_loss:.6f}")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    @torch.no_grad()
    def save_checkpoint(self, val_loss, model):
        self.state_dict = deepcopy(model.state_dict())
        self.val_loss_min = val_loss

    @torch.no_grad()
    def load_checkpoint(self, model):
        model.load_state_dict(self.state_dict)
