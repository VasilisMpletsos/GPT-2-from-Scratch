import math


class CosineLearningDecay:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        optimizer=optimizer,
        max_steps: int = 100,
        warmup_steps: int = 0,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer = optimizer

    def update_lr(self, step: int):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        if step > self.max_steps:
            return self.min_lr

        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)

        coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        step_lr = self.min_lr + coefficient * (self.max_lr - self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = step_lr
        return step_lr
