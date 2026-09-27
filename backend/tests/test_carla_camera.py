import numpy as np
import pytest
from app.pipeline.carom_geometry import RoadCamera

def config():
    return {'planar_calibration':{'method':'carla_exact_extrinsics',
        'intrinsic_matrix':[[600,0,640],[0,600,360],[0,0,1]],
        'world_to_camera_cv':[[1,0,0,0],[0,-1,0,0],[0,0,-1,20],[0,0,0,1]],'distortion':[0]*5}}

def test_simulator_projection_roundtrip_and_provenance():
    c,a=RoadCamera.from_scene(config())
    xy=np.array([[0,0],[4,10],[-4,10]])
    assert np.allclose(c.ground(c.project(np.c_[xy,np.zeros(3)])),xy)
    assert a['status']=='simulation_parameters_provided'
    assert not a['surveyed'] and a['control_reprojection_rmse_px'] is None

def test_reject_left_handed_rotation_and_distortion():
    cfg=config();cfg['planar_calibration']['world_to_camera_cv'][0][0]=-1
    with pytest.raises(ValueError):RoadCamera.from_scene(cfg)
    cfg=config();cfg['planar_calibration']['distortion'][0]=.1
    with pytest.raises(ValueError):RoadCamera.from_scene(cfg)
