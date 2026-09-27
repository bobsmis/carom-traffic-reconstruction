import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('carla_eval',Path(__file__).resolve().parents[2]/'scripts/evaluate_carla_proxy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def gt(actor=1):
    return {'actor_id':actor,'in_camera_frustum':True,'projected_cuboid_px':[[10,10],[30,10],[30,30],[10,30]]}

def test_match_does_not_use_metric_predictions():
    p={'bbox_xyxy':[10,10,30,30],'ground_xy':[9999,-9999]}
    matches,n=module.associate([p],[gt()],100,100)
    assert len(matches)==1 and n==1

def test_ambiguous_matches_are_not_scored():
    p={'bbox_xyxy':[10,10,30,30]}
    assert module.associate([p],[gt(1),gt(2)],100,100)[0]==[]

def test_unmatched_and_empty():
    assert module.associate([], [gt()],100,100)==([],1)
    assert module.associate([{'bbox_xyxy':[70,70,90,90]}],[gt()],100,100)[0]==[]
