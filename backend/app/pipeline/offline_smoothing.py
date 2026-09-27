"""Offline constant-velocity RTS inference from raw accepted position evidence.

This is an explicitly labelled engineering extension, not CAROM source code.
No lane snapping, collision relocation, synthetic detections or identity merging.
"""
from __future__ import annotations
import numpy as np


def transition(dt, acceleration_spectral_density=2.25):
    if not np.isfinite(dt) or dt<=0 or acceleration_spectral_density<=0:
        raise ValueError('positive dt and process noise required')
    f=np.eye(4); f[:2,2:]=np.eye(2)*dt
    q=np.block([[np.eye(2)*dt**3/3,np.eye(2)*dt**2/2],
                [np.eye(2)*dt**2/2,np.eye(2)*dt]])*acceleration_spectral_density
    return f,q


def smooth_positions(times, observations, variances, acceleration_spectral_density=2.25):
    """None is missing evidence, never a zero coordinate or repeated measurement.

    Caller supplies a contiguous lifecycle segment starting with an observation.
    Returns posterior states/covariances and actual per-observation gate results.
    Initial velocity remains uncertain, not inferred from any lane direction.
    """
    times=np.asarray(times,float); variances=np.asarray(variances,float)
    n=len(times)
    if n<1 or len(observations)!=n or len(variances)!=n or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid timeline')
    if observations[0] is None or not np.isfinite(variances).all() or np.any(variances<=0):
        raise ValueError('initial position and positive finite variances required')
    for z in observations:
        if z is not None and (np.asarray(z).shape!=(2,) or not np.isfinite(z).all()):
            raise ValueError('invalid position evidence')
    h=np.c_[np.eye(2),np.zeros((2,2))]
    x=np.r_[observations[0],0.,0.].astype(float)
    p=np.diag([variances[0],variances[0],100.,100.])
    filtered=[]; covariance=[]; predicted=[]; predicted_covariance=[]; transitions=[]; accepted=[]
    for i,z in enumerate(observations):
        f=np.eye(4)
        if i:
            f,q=transition(times[i]-times[i-1],acceleration_spectral_density)
            x=f@x; p=f@p@f.T+q
        predicted.append(x.copy()); predicted_covariance.append(p.copy()); transitions.append(f)
        used=i==0
        if i and z is not None:
            r=np.eye(2)*variances[i]; innovation=np.asarray(z)-h@x; s=h@p@h.T+r
            if innovation@np.linalg.solve(s,innovation)<=16.:
                k=np.linalg.solve(s,h@p).T; x=x+k@innovation
                ikh=np.eye(4)-k@h
                p=ikh@p@ikh.T+k@r@k.T; used=True
        filtered.append(x.copy()); covariance.append(p.copy()); accepted.append(used)
    xs=np.array(filtered); ps=np.array(covariance)
    for i in range(n-2,-1,-1):
        gain=np.linalg.solve(predicted_covariance[i+1],transitions[i+1]@covariance[i]).T
        xs[i]=filtered[i]+gain@(xs[i+1]-predicted[i+1])
        ps[i]=covariance[i]+gain@(ps[i+1]-predicted_covariance[i+1])@gain.T
        ps[i]=(ps[i]+ps[i].T)/2
    return xs,ps,accepted
