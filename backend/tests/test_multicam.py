import numpy as np
from app.pipeline.multicam import GlobalTracker

def obs(camera,local=1,x=0,y=125):
    return dict(camera=camera,local_id=local,xy=np.array([x,y],float),velocity=np.array([0,5.]),
        appearance=np.array([1.,0.]),heading=1.57,class_name='car',variance=1.,dimensions=[4.5,1.8,1.5],observed=True)

def test_overlap_one_global_id():
    t=GlobalTracker();o=t.update([obs('a'),obs('b')],0)
    assert len(o)==1 and o[0]['cameras']==['a','b']

def test_adjacent_lanes_not_merged():
    t=GlobalTracker();assert len(t.update([obs('a'),obs('b',x=3.5)],0))==2

def test_handoff_preserves_id_without_oracle():
    t=GlobalTracker();first=t.update([obs('a')],0)[0]['id']
    assert t.update([obs('b',y=130)],1)[0]['id']==first

def test_ambiguity_and_timeout():
    t=GlobalTracker();t.update([obs('a',1,x=-.2),obs('a',2,x=.2)],0)
    assert t.update([obs('b')],0)[0]['id']==3
    assert t.update([obs('b')],5)[0]['id']==4
