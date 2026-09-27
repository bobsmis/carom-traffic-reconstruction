"""Per-identity rigid dimensions; pixel bbox changes are not physical resizing."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleSize:
    required_samples: int = 12
    max_step_fraction: float = .01
    value: np.ndarray | None = None
    target: np.ndarray | None = None
    samples: list = field(default_factory=list)
    locked: bool = False

    def observe(self, fit):
        """Call on a proposed copy; commit only after geometry/Kalman acceptance.

        A provisional first-fit shape stays fixed while gathering evidence.
        A robust multi-frame estimate replaces it gradually, then freezes for
        the lifetime of the ID. No resizing on misses or class-label changes.
        """
        if self.locked or not fit or fit.get('status') != 'fitted':
            return
        dims = np.asarray(fit['dimensions_lwh'], float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            return
        if self.value is None:
            self.value = dims.copy()
        reliable = (fit.get('silhouette_hull_iou', 0) >= .75
                    and fit.get('support_rmse_m', float('inf')) <= .20
                    and fit.get('support_rank', 0) >= 5)
        if reliable and self.target is None:
            self.samples.append(dims.tolist())
            self.samples = self.samples[-36:]
            if len(self.samples) >= self.required_samples:
                estimates = np.asarray(self.samples)
                median = np.median(estimates, axis=0)
                # Reject isolated large dimension estimates; consistent units are metres.
                inliers = np.max(np.abs(estimates / median - 1), axis=1) <= .15
                if inliers.sum() >= self.required_samples:
                    self.target = np.median(estimates[inliers], axis=0)
        if self.target is not None:
            delta = self.target - self.value
            limit = self.value * self.max_step_fraction
            if np.all(np.abs(delta) <= limit):
                self.value = self.target.copy()
                self.locked = True
            else:
                self.value += np.clip(delta, -limit, limit)

    def report(self):
        return {'state': 'locked' if self.locked else 'settling' if self.target is not None else 'provisional',
                'accepted_size_samples': len(self.samples), 'required_samples': self.required_samples,
                'max_step_fraction': self.max_step_fraction,
                'target_lwh': self.target.tolist() if self.target is not None else None,
                'method': 'per_id_robust_initialization_then_rigid_size',
                'absolute_size_accuracy_validated': False}
