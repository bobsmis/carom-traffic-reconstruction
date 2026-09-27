import numpy as np
import pytest
from app.pipeline.offline_smoothing import smooth_positions,transition


def test_rts_reduces_noise_and_preserves_constant_velocity_through_dropout():
    rng=np.random.default_rng(12); t=np.arange(100)*.05
    truth=np.c_[2+t*.2,10+t*8]
    z=truth+rng.normal(0,.3,truth.shape)
    observations=list(z.copy()); observations[35:45]=[None]*10
    states,cov,accepted=smooth_positions(t,observations,np.full(len(t),.09))
    assert np.sqrt(np.mean((states[:,:2]-truth)**2))<np.sqrt(np.mean((z-truth)**2))*.6
    assert np.max(np.linalg.norm(states[35:45,:2]-truth[35:45],axis=1))<.5
    assert np.allclose(np.median(states[:,2:],axis=0),[.2,8],atol=.2)
    assert not any(accepted[35:45])
    assert np.linalg.eigvalsh(cov).min()>0


def test_extreme_observation_rejected_and_stationary_not_moved():
    t=np.arange(50)*.1; z=[[3.,4.] for _ in t];z[25]=[200.,-300.]
    states,_,accepted=smooth_positions(t,z,np.full(len(t),.1))
    assert not accepted[25]
    assert np.allclose(states[:,:2],[3,4])


def test_turn_is_not_snapped_to_straight_lane():
    t=np.arange(150)*.05
    truth=np.c_[.12*t*t,10+4*t]
    xs,_,_=smooth_positions(t,list(truth),np.full(len(t),.04))
    assert xs[-1,0]-xs[0,0]>5
    assert xs[-1,2]>xs[10,2]+1
    assert np.sqrt(np.mean((xs[:,:2]-truth)**2))<.15


@pytest.mark.parametrize('times,obs,var',[
    ([0,0],[[0,0],[0,0]],[1,1]),([0,1],[None,[0,0]],[1,1]),
    ([0],[[float('nan'),1]],[1]),([0],[[0,0]],[-1])])
def test_bad_inputs_rejected(times,obs,var):
    with pytest.raises(ValueError): smooth_positions(times,obs,var)


def test_process_covariance_positive():
    for dt in (.001,.1,1.):
        _,q=transition(dt)
        assert np.linalg.eigvalsh(q).min()>0
