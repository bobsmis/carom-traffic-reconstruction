import numpy as np
import pytest

from app.pipeline.metric_bev import MetricViewport
from app.pipeline.carom_geometry import cuboid_vertices


def viewport(scale=20):
    return MetricViewport(scale,-20,20,-4,104)


@pytest.mark.parametrize('xy', [[0,0],[-12.25,50],[12.25,99]])
def test_one_meter_is_twenty_pixels_in_both_axes_at_all_distances(xy):
    v=viewport(); p=v.project([xy,[xy[0]+1,xy[1]],[xy[0],xy[1]+1]])
    assert np.linalg.norm(p[1]-p[0])==pytest.approx(20)
    assert np.linalg.norm(p[2]-p[0])==pytest.approx(20)


def test_double_scale_doubles_distances_not_world_coordinates():
    world=np.array([[2.,8.],[2.,40.]])
    before=world.copy()
    old=viewport(10).project(world); new=viewport(20).project(world)
    assert np.linalg.norm(new[1]-new[0])==2*np.linalg.norm(old[1]-old[0])
    assert np.array_equal(world,before)


@pytest.mark.parametrize('heading',[0,.7,1.57,-2.3])
def test_rotated_car_keeps_metric_aspect_ratio(heading):
    polygon=viewport().project(cuboid_vertices([4.,70.],heading,[4.5,1.8,1.5])[:4,:2])
    lengths=sorted(np.linalg.norm(np.roll(polygon,-1,axis=0)-polygon,axis=1))
    assert np.allclose(lengths,[36,36,90,90])


def test_whole_roi_fits_native_canvas_without_one_axis_stretch():
    v=viewport(); p=v.project([[-18,0],[18,0],[18,100],[-18,100]])
    assert v.width==2160 and v.height==800
    assert np.all(p>=0) and np.all(p[:,0]<v.width) and np.all(p[:,1]<v.height)
    assert np.allclose(v.transform[:2,:2].T@v.transform[:2,:2],np.eye(2)*400)


def test_invalid_scale_rejected():
    with pytest.raises(ValueError): viewport(0)
    with pytest.raises(ValueError): viewport(float('nan'))
