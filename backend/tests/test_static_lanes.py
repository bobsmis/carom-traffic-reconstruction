import pytest
from app.pipeline.static_lanes import lane_membership

LANES=[{'road_id':1,'lane_id':1,'center_lateral_m':-1.75,'width_m':3.5,'forward_road_xy':[0,-1]},
       {'road_id':1,'lane_id':-1,'center_lateral_m':1.75,'width_m':3.5,'forward_road_xy':[0,1]}]

def test_signed_lanes_and_no_snap():
    r=lane_membership(1.2,20,LANES,[0,100])
    assert r['lane_id']==-1
    assert r['lateral_offset_m']==pytest.approx(-.55)
    assert not r['position_was_snapped']

def test_boundary_is_not_forced_to_lane():
    assert lane_membership(0,20,LANES,[0,100])['lane_id'] is None
    assert lane_membership(2,120,LANES,[0,100])['status']=='outside_static_map'

def test_direction_diagnostic_does_not_change_lane():
    r=lane_membership(1,20,LANES,[0,100],heading=-1.57079632679)
    assert r['lane_id']==-1
    assert r['direction_difference_rad']>3.14
