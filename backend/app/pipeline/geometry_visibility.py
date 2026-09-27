"""Bounded metric prediction, independent of detector-ID retirement.

Uncertainty is conditional on the configured camera, not calibration accuracy.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeometryVisibility:
    max_gap_seconds: float = 2.
    max_position_sigma_m: float = 2.5

    def __post_init__(self):
        if not np.isfinite([self.max_gap_seconds,self.max_position_sigma_m]).all() or min(self.max_gap_seconds,self.max_position_sigma_m)<=0:
            raise ValueError('invalid geometry prediction limits')

    def evaluate(self, metric, geometry_age, detector_lost, dt, max_lost, region, *, accepted_positions=0):
        if not np.isfinite(dt) or dt<=0 or geometry_age<0 or detector_lost<0:
            raise ValueError('invalid prediction age')
        out={'publish':False,'prediction_age_s':float(geometry_age*dt),
             'max_gap_seconds':self.max_gap_seconds,'max_position_sigma_m':self.max_position_sigma_m,
             'uncertainty_scope':'conditional_on_unverified_camera','accepted_positions':accepted_positions}
        if metric is None:
            return dict(out,reason='geometry_not_initialized')
        if detector_lost>max_lost:
            return dict(out,reason='detector_lifecycle_expired')
        # Beyond the short detector-coast window require repeated metric
        # evidence, not one cuboid and an essentially uninitialized velocity.
        if geometry_age>max_lost and accepted_positions<3:
            return dict(out,reason='insufficient_motion_evidence_for_extension')
        if geometry_age*dt>self.max_gap_seconds:
            return dict(out,reason='geometry_prediction_timeout')
        cov=np.asarray(metric.covariance[:2,:2],float)
        if not np.isfinite(metric.mean).all() or not np.isfinite(cov).all():
            return dict(out,reason='nonfinite_prediction')
        eig=np.linalg.eigvalsh((cov+cov.T)/2)
        if eig.min() < -1e-8:
            return dict(out,reason='invalid_prediction_covariance')
        sigma=float(np.sqrt(max(0,eig.max())))
        out['position_sigma_m']=sigma
        if sigma>self.max_position_sigma_m:
            return dict(out,reason='prediction_uncertainty_exceeded')
        gate=region.ground_check(metric.mean[:2]);out['ground_gate']=gate
        if not gate['eligible']:
            return dict(out,reason=gate['reason'])
        return dict(out,publish=True,reason='observed_geometry' if geometry_age==0 else 'bounded_motion_prediction')
