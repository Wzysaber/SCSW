import torch
import torch.optim as optim
import numpy as np


class PolyLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
            for base_lr in self.base_lrs
        ]


class CosWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None,
                 **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8, )

        self.global_step = 0
        self.warmup_iter = np.float(warmup_iter)
        self.warmup_ratio = warmup_ratio
        self.max_iter = np.float(max_iter)
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = self.global_step / self.warmup_iter
            lr_add = (1 - self.global_step / self.warmup_iter) * self.warmup_ratio
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult + lr_add

        elif self.global_step < self.max_iter:

            lr_mult = np.cos(
                (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter) * np.pi) * 0.5 + 0.5
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1


class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None,
                 **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8, )

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter:

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1


class PolyWarmupSGD(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None,
                 **kwargs):
        super().__init__(params, lr=lr, momentum=0.9, weight_decay=weight_decay, )

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_lr = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = (1 - self.global_step / self.warmup_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult * 10

        elif self.global_step < self.max_iter:

            lr_mult = (1 - (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1
