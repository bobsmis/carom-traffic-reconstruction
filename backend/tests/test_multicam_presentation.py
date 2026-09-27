import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from process_multicam import volume_verdict,map_view

def valid():
    return {'dimensions_lwh':[4.5,1.8,1.5],'fit_quality':{'status':'fitted','silhouette_hull_iou':.85},
            'size_stabilization':{'state':'locked'}}

def test_rigid_supported_volume():
    assert volume_verdict(valid())['accepted']

def test_bad_width_is_not_silently_clamped():
    obj=valid();obj['dimensions_lwh'][1]=5
    assert not volume_verdict(obj)['accepted']
    assert obj['dimensions_lwh'][1]==5

def test_provisional_fit_marked_uncertain():
    obj=valid();obj['size_stabilization']['state']='provisional'
    assert 'size_not_locked' in volume_verdict(obj)['reasons']

def test_missing_geometry_has_reasons():
    assert not volume_verdict({})['accepted']

def test_crop_one_road_is_isotropic():
    image=map_view([],[],0)
    assert image.shape==(540,960,3)
    # Longitudinal 60 m uses 840 px, road 28 m uses 392 px, same 14 px/m.
    assert np.array_equal(image[66,39],[62,67,72])
    assert np.array_equal(image[456,877],[62,67,72])
