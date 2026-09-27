import math

import cv2
import numpy as np
import pytest

from app.pipeline.carom_geometry import RoadCamera, cuboid_vertices, fit_cuboid, flow_heading, flow_velocity, masked_flow


def camera():
    origin = np.array([0., -20., 12.])
    z = np.array([0., 50., -12.]); z /= np.linalg.norm(z)
    x = np.cross(z, [0.,0.,1.]); x /= np.linalg.norm(x)
    R = np.vstack((x,np.cross(z,x),z))
    return RoadCamera(np.array([[1200.,0,960],[0,1200.,540],[0,0,1]]), R, -R@origin)


def test_ground_and_cuboid_share_projection():
    c = camera()
    xy = np.array([[1.,10.],[-7.,70.],[6.,30.]])
    assert np.allclose(c.ground(c.project(np.c_[xy,np.zeros(3)])), xy)
    assert np.linalg.det(c.R) == pytest.approx(1)


@pytest.mark.parametrize('heading', [math.pi/2, -.9, .3, 0.])
def test_silhouette_tangents_recover_known_cuboid(heading):
    c = camera()
    xy, dims = [3.,28.], [4.6,1.9,1.6]
    corners = c.project(cuboid_vertices(xy,heading,dims))
    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1,2)
    fit = fit_cuboid(c,hull,heading,[4.5,1.8,1.5])
    assert fit['status'] == 'fitted', fit
    assert np.allclose(fit['ground_xy'],xy,atol=.1)
    assert np.allclose(fit['dimensions_lwh'],dims,atol=.12)
    assert fit['silhouette_hull_iou'] > .98


@pytest.mark.parametrize('direction', [[0.,1.],[0.,-1.],[1.,0.]])
def test_heading_with_outliers_and_vanishing_point_at_infinity(direction):
    c = camera(); rng = np.random.default_rng(5)
    points = rng.uniform([-2,25,.3],[2,30,1.5],(30,3))
    d = np.r_[direction,0.]
    f = np.c_[c.project(points),c.project(points + d*.2)]
    f[-8:,2:] += rng.uniform(-20,20,(8,2))
    result = flow_heading(c,f)
    assert result is not None
    actual = np.array([math.cos(result['heading_rad']),math.sin(result['heading_rad'])])
    assert actual @ np.array(direction) > .999
    assert result['inlier_count'] >= 22


def test_stationary_flow_not_given_an_invented_heading():
    assert flow_heading(camera(),np.tile([400.,500.,400.,500.],(20,1))) is None


def test_invalid_silhouette_rejected():
    assert fit_cuboid(camera(),[[1,1],[2,2],[3,3]],0,[4.5,1.8,1.5])['status'] == 'rejected'


def test_strict_calibration_rejects_nominal_four_points():
    c = camera(); xy = [[-10,0],[10,0],[-10,80],[10,80]]
    cfg = {'crop':{'width':1920,'height':1080}, 'planar_calibration':{
        'focal_length_px':1200., 'ground_points_m':xy,
        'image_points_px':c.project(np.c_[xy,np.zeros(4)]).tolist()}}
    with pytest.raises(ValueError,match='strict calibration'):
        RoadCamera.from_scene(cfg)
    fitted,audit = RoadCamera.from_scene(cfg,experimental=True)
    assert audit['control_reprojection_rmse_px'] < .001
    assert audit['status'] == 'experimental_unverified'
    assert np.allclose(fitted.P/fitted.P[-1,-1],c.P/c.P[-1,-1],atol=1e-4)


def test_masked_flow_translation():
    rng = np.random.default_rng(42)
    a = np.zeros((180,240),np.uint8)
    a[50:130,60:170] = rng.integers(0,255,(80,110),dtype=np.uint8)
    b = cv2.warpAffine(a,np.float32([[1,0,3],[0,1,2]]),(240,180))
    f = masked_flow(a,b,[[60,50],[170,50],[170,130],[60,130]])
    assert len(f) >= 10
    assert np.allclose(np.median(f[:,2:]-f[:,:2],axis=0),[3,2],atol=.15)


def test_flow_velocity_on_visible_cuboid_surface():
    c = camera(); xy=[3.,28.]; dims=[4.6,1.9,1.6]; heading=math.pi/2
    # Near end face y=28-L/2, known rigid translation in y.
    points=np.array([[x,28-dims[0]/2,z] for x in np.linspace(2.2,3.8,5) for z in np.linspace(.3,1.4,5)])
    start=c.project(points); end=c.project(points+[0.,.1,0.])
    fit={'ground_xy':xy,'dimensions_lwh':dims,'heading_rad':heading}
    velocity=flow_velocity(c,fit,np.c_[start,end],.02)
    assert velocity is not None
    assert np.allclose(velocity['xy_mps'],[0,5],atol=.08)


def surveyed_config():
    c=camera(); xy=np.array([[-10,0],[0,0],[10,0],[-10,80],[0,80],[10,80]],float)
    hold=np.array([[-4,25],[5,55]],float)
    return {'crop':{'width':1920,'height':1080},'planar_calibration':{
        'intrinsic_matrix':c.K.tolist(),'surveyed':True,'ground_points_m':xy.tolist(),
        'image_points_px':c.project(np.c_[xy,np.zeros(len(xy))]).tolist(),
        'check_points':{'ground_points_m':hold.tolist(),'image_points_px':c.project(np.c_[hold,np.zeros(2)]).tolist()}}}


def test_independent_survey_checks_pass_for_synthetic_known_camera():
    _,audit=RoadCamera.from_scene(surveyed_config())
    assert audit['status']=='calibration_check_passed'
    assert audit['check_rmse_px']<1e-4


def test_controls_relabelled_as_holdouts_are_rejected():
    cfg=surveyed_config(); c=cfg['planar_calibration']
    c['check_points']={key:c[key][:2] for key in ('ground_points_m','image_points_px')}
    with pytest.raises(ValueError,match='reuse'):
        RoadCamera.from_scene(cfg)


def test_invalid_intrinsics_and_collinear_controls_are_rejected():
    cfg=surveyed_config(); cfg['planar_calibration']['intrinsic_matrix'][0][0]=-1
    with pytest.raises(ValueError,match='intrinsic'): RoadCamera.from_scene(cfg)
    cfg=surveyed_config();cfg['planar_calibration']['ground_points_m']=[[0,i] for i in range(6)]
    with pytest.raises(ValueError,match='collinear'): RoadCamera.from_scene(cfg)
