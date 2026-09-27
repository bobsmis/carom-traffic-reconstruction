import copy

import numpy as np
import pytest

from app.pipeline.carom_geometry import RoadCamera
from app.pipeline.carom_tracking import MetricState
from app.pipeline.measurement_region import MeasurementRegion


def setup_region():
    origin=np.array([0.,-20.,12.])
    z=np.array([0.,50.,-12.]); z/=np.linalg.norm(z)
    x=np.cross(z,[0.,0.,1.]); x/=np.linalg.norm(x)
    R=np.vstack((x,np.cross(z,x),z))
    camera=RoadCamera(np.array([[1200.,0,960],[0,1200.,540],[0,0,1]]),R,-R@origin)
    settings={'enabled':True,'ground_polygon_m':[[-14,0],[14,0],[14,50],[-14,50]],
              'max_camera_ground_distance_m':75.,'min_bbox_width_px':24.,'min_bbox_height_px':24.,
              'min_mask_area_px':400.,'pixel_noise_sigma_px':2.,'max_projection_sigma_m':.75,
              'detector_context_px':96}
    return MeasurementRegion(camera,{'measurement_region':settings})


def observation(region,xy,width=40.,height=40.):
    u,v=region.camera.project([[*xy,0.]])[0]
    return {'bbox_xyxy':[u-width/2,v-height,u+width/2,v],
            'mask_polygon':[[u-width/2,v-height],[u+width/2,v-height],[u+width/2,v],[u-width/2,v]]}


def test_disabled_region_preserves_old_behavior():
    r=MeasurementRegion(None,{})
    assert r.ground_check([1000,1000])['eligible']
    assert r.observation_check({})['eligible']
    assert r.detector_crop((1080,1920,3))==(0,0,1920,1080)


def test_near_vehicle_and_preselection_not_a_measurement():
    r=setup_region(); result=r.observation_check(observation(r,[2,30]))
    assert result['eligible']
    assert result['probe_role']=='preselection_only_not_vehicle_position'
    assert 'ground_xy' not in result


@pytest.mark.parametrize('xy,reason', [([0,100],'beyond_camera_range'),
                                     ([15,20],'outside_measurement_region'),
                                     ([0,-1],'outside_measurement_region')])
def test_range_and_lateral_limits(xy,reason):
    assert setup_region().ground_check(xy)['reason']==reason


def test_small_mask_and_small_bbox_are_rejected():
    r=setup_region()
    assert r.observation_check(observation(r,[0,20],width=23))['reason']=='insufficient_pixel_extent'
    d=observation(r,[0,20]); d['mask_polygon']=[[960,600],[962,602],[960,604]]
    assert r.observation_check(d)['reason']=='insufficient_mask_area'


def test_detector_context_edge_truncation_is_not_used_for_measurement():
    r=setup_region(); d=observation(r,[0,20]); d['detector_crop_truncated']=True
    assert r.observation_check(d)['reason']=='detector_crop_truncation'


def test_nonfinite_observation_and_bad_configuration():
    r=setup_region(); d=observation(r,[0,20]); d['bbox_xyxy'][0]=float('nan')
    assert r.observation_check(d)['reason']=='invalid_observation'
    cfg=copy.deepcopy(r.settings); cfg['min_bbox_width_px']=-1
    with pytest.raises(ValueError,match='threshold'):
        MeasurementRegion(r.camera,{'measurement_region':cfg})


def test_pixel_noise_sensitivity_increases_with_distance():
    r=setup_region()
    close=r.ground_check([0,10])['conditional_projection_sigma_m']
    far=r.ground_check([0,45])['conditional_projection_sigma_m']
    assert close < far
    r.settings['max_projection_sigma_m']=(close+far)/2
    assert r.ground_check([0,10])['eligible']
    assert r.ground_check([0,45])['reason']=='projection_too_sensitive'


def test_crop_is_bounded_and_contains_projected_region_with_context():
    r=setup_region(); x1,y1,x2,y2=r.detector_crop((1080,1920,3))
    assert 0<=x1<x2<=1920 and 0<y1<y2<=1080
    p=r.image_polygon()
    assert y1<=p[:,1].min() and y2>=min(1080,p[:,1].max())
    uv=np.array([50.,30.])+[x1,y1]
    assert np.allclose(uv-[x1,y1],[50,30])


def test_prediction_outside_region_is_not_eligible():
    r=setup_region(); state=MetricState.at([0,49.9]); state.mean[3]=10
    assert r.ground_check(state.mean[:2])['eligible']
    state.predict(.1)
    assert not r.ground_check(state.mean[:2])['eligible']


def test_split_conserves_candidates_and_does_not_change_coordinates():
    r=setup_region(); items=[observation(r,[0,20]),observation(r,[0,100])]
    before=copy.deepcopy(items); kept,rejected=r.split(items)
    assert len(kept)+len(rejected)==len(items)
    assert kept[0] is items[0] and items==before
    assert rejected[0]['candidate_index']==1
