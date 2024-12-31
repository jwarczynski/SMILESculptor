from typing import Optional

from lightning.pytorch.callbacks import EarlyStopping
from torch import Tensor


class EarlyStoppingExt(EarlyStopping):

    def __init__(
            self,
            monitor: str,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = False,
            mode: str = "min",
            strict: bool = True,
            check_finite: bool = True,
            stopping_threshold: Optional[float] = None,
            divergence_threshold: Optional[float] = None,
            check_on_train_epoch_end: Optional[bool] = None,
            log_rank_zero_only: bool = False,
            reset_on_improvement: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only
        )
        self.reset_on_improvement = reset_on_improvement
        self.last_value = self.best_score
        self.checks_without_improvement = 0

    def _evaluate_stopping_criteria(self, current: Tensor) -> tuple[bool, Optional[str]]:
        """
        Evaluate the stopping criteria. If the monitored metric has improved since the last check,
        update the last value and reset the patience counter. If the monitored metric has not improved since last check,
        increment the patience counter and check if training should stop.

        Args:
        - current (torch.Tensor): The current value of the monitored metric

        Returns:
        - stop (bool): Whether to stop training
        - reason (str): The reason for stopping training
        """

        # Evaluate the stopping criteria
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        if self.reset_on_improvement:
            improve = self.monitor_op(current - self.min_delta, self.last_value)

            # should stop beacuse of no improvement w.r.t. best score but improvement from last eval
            # if should stop from another reason do not continue
            if should_stop and self.wait_count >= self.patience:
                should_stop = not improve
                reason = f"no improvement in monitored metric since last {self.patience} steps" if not improve else reason

            self.wait_count = 0 if improve else self.wait_count
            self.last_value = current

        return should_stop, reason
