import numpy as np
from app.pipeline.geometry_visibility import GeometryVisibility
from app.pipeline.carom_tracking import MetricState


class Region:
    def ground_check(self,xy):
        return {'eligible':abs(xy[0])<18,'reason':'outside_measurement_region'}


def test_fit_dropout_independent_of_detector_retirement():
    state=MetricState.at([1.,50.]);state.covariance=np.eye(4)*.1
    policy=GeometryVisibility()
    # ID 8 regression: visible detection, 20 frames since last accepted cuboid.
    assert policy.evaluate(state,20,0,1/30,18,Region(),accepted_positions=3)['publish']
    assert policy.evaluate(state,20,18,1/30,18,Region(),accepted_positions=3)['publish']
    assert not policy.evaluate(state,20,19,1/30,18,Region())['publish']


def test_prediction_is_bounded_and_never_initializes_unknown_position():
    policy=GeometryVisibility();state=MetricState.at([1.,50.])
    assert not policy.evaluate(None,0,0,1/30,18,Region())['publish']
    assert policy.evaluate(state,61,0,1/30,18,Region(),accepted_positions=3)['reason']=='geometry_prediction_timeout'
    state.covariance[:2,:2]=np.eye(2)*9
    assert policy.evaluate(state,20,0,1/30,18,Region(),accepted_positions=3)['reason']=='prediction_uncertainty_exceeded'
    state.covariance=np.eye(4);state.mean[0]=19
    assert not policy.evaluate(state,10,0,1/30,18,Region())['publish']


def test_single_fit_does_not_authorize_extended_motion_prediction():
    result=GeometryVisibility().evaluate(MetricState.at([1.,50.]),19,0,1/30,18,Region(),accepted_positions=1)
    assert not result['publish'] and result['reason']=='insufficient_motion_evidence_for_extension'
