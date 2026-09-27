from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline.vehicle_size import VehicleSize
from app.pipeline.carom_geometry import RoadCamera, fit_cuboid, cuboid_vertices
from app.pipeline.measurement_region import MeasurementRegion


def observation(dims, **kwargs):
    return dict(status='fitted',dimensions_lwh=dims,silhouette_hull_iou=.9,
                support_rmse_m=.05,support_rank=5,**kwargs)


def test_locked_shape_ignores_later_noise_and_class_size_changes():
    size=VehicleSize()
    for _ in range(12): size.observe(observation([4.6,1.85,1.5]))
    assert size.locked
    before=size.value.copy()
    for dims in ([8.,2.5,3.2],[2.7,1.2,.9],[5.2,2.2,1.8]):
        size.observe(observation(dims))
        assert np.array_equal(size.value,before)


def test_provisional_constant_then_bounded_settle():
    size=VehicleSize()
    size.observe(observation([5.,2.,1.6]))
    before=size.value.copy()
    for _ in range(10):
        size.observe(observation([4.6,1.85,1.5]))
        assert np.array_equal(size.value,before)
    for _ in range(40):
        before=size.value.copy()
        size.observe(observation([4.6,1.85,1.5]))
        assert np.max(np.abs(size.value/before-1))<=.010000001
    assert size.locked and np.allclose(size.value,[4.6,1.85,1.5])


def test_low_quality_and_missing_observations_do_not_lock_or_resize():
    size=VehicleSize(); size.observe(observation([4.6,1.85,1.5])); before=size.value.copy()
    for _ in range(25):
        low=observation([7.,3.,4.]); low['silhouette_hull_iou']=.5
        size.observe(low); size.observe(None)
    assert not size.locked and len(size.samples)==1
    assert np.array_equal(before,size.value)


def test_rejected_state_proposal_has_no_side_effect_and_ids_are_independent():
    a=VehicleSize(); a.observe(observation([4.6,1.85,1.5]))
    proposal=deepcopy(a)
    for _ in range(20): proposal.observe(observation([4.6,1.85,1.5]))
    assert proposal.locked and not a.locked and len(a.samples)==1
    b=VehicleSize(); b.observe(observation([12.,2.5,3.2]))
    assert not np.array_equal(a.value,b.value)


def test_fixed_dimensions_refit_position_without_rescaling():
    origin=np.array([0.,-20.,12.]); z=np.array([0.,50.,-12.]); z/=np.linalg.norm(z)
    x=np.cross(z,[0,0,1]); x/=np.linalg.norm(x); R=np.vstack((x,np.cross(z,x),z))
    camera=RoadCamera(np.array([[1200.,0,960],[0,1200.,540],[0,0,1]]),R,-R@origin)
    dims=[4.6,1.85,1.5]
    for center in ([3.,28.],[4.,31.],[5.,34.]):
        hull=cv2.convexHull(camera.project(cuboid_vertices(center,1.2,dims)).astype(np.float32)).reshape(-1,2)
        fit=fit_cuboid(camera,hull,1.2,[4.5,1.8,1.5],fixed_dimensions=dims)
        assert fit['status']=='fitted' and fit['dimensions_fixed']
        assert np.allclose(fit['ground_xy'],center,atol=.001)
        assert np.array_equal(fit['dimensions_lwh'],dims)
    with pytest.raises(ValueError):
        fit_cuboid(camera,hull,1.2,dims,fixed_dimensions=[-1,2,3])


def test_current_expanded_scene_admits_all_eight_lane_centers_to_100m():
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'configs/user_bridge_traffic.json').read_text())
    camera,_=RoadCamera.from_scene(config,experimental=True)
    region=MeasurementRegion(camera,config)
    assert np.ptp(region.polygon[:,0])==36
    assert np.ptp(region.polygon[:,1])==100
    for y in (0,25,50,75,100):
        for x in np.arange(-12.25,12.26,3.5):
            result=region.ground_check([x,y])
            assert result['eligible'], (x,y,result)
