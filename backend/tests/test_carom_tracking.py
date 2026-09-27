import numpy as np
from app.pipeline.carom_tracking import FlowMaskTracker, MetricState, deduplicate_observations


def detection(x):
    return {'class_name':'car','mask_polygon':[[x,20],[x+30,20],[x+30,50],[x,50]]}


def test_coast_reassociate_and_nonrecycled_id():
    tracker=FlowMaskTracker(max_lost=2); gray=np.zeros((120,120),np.uint8)
    tracks,_=tracker.update([detection(20)],None,gray,.1)
    assert tracks[0].id==1
    tracker.update([],gray,gray,.1)
    tracks,_=tracker.update([detection(22)],gray,gray,.1)
    assert tracks[0].id==1 and tracks[0].lost==0
    tracker.update([],gray,gray,.1); tracker.update([],gray,gray,.1)
    tracks,retired=tracker.update([],gray,gray,.1)
    assert not tracks and retired==[1]
    tracks,_=tracker.update([detection(22)],gray,gray,.1)
    assert tracks[0].id==2


def test_kalman_velocity_coast_and_outlier_gate():
    k=MetricState.at([1,2]); assert k.observe([0,10],velocity=True,variance=.01)
    k.predict(.5)
    assert np.allclose(k.mean[:2],[1,7],atol=.01)
    before=k.mean.copy()
    assert not k.observe([900,900],variance=.1)
    assert np.array_equal(before,k.mean)
    assert np.linalg.eigvalsh(k.covariance).min()>0


def test_one_to_one_assignment():
    tracker=FlowMaskTracker(); gray=np.zeros((120,140),np.uint8)
    tracker.update([detection(20),detection(80)],None,gray,.1)
    tracks,_=tracker.update([detection(22),detection(82)],gray,gray,.1)
    assert [t.id for t in tracks]==[1,2]
    assert len({id(t.detection) for t in tracks})==2


def test_classification_jitter_does_not_create_coasting_duplicate():
    tracker=FlowMaskTracker(); gray=np.zeros((120,140),np.uint8)
    for x,cls in zip([20,21,22,23,24],['bus','bus','bus','truck','car']):
        d=detection(x); d['class_name']=cls
        tracks,_=tracker.update([d],None,gray,.1)
        assert len(tracks)==1 and tracks[0].id==1 and tracks[0].lost==0
    assert tracks[0].class_name=='bus'


def test_duplicate_classes_do_not_create_two_identities_for_one_mask():
    a=detection(20); a.update(class_name='bus',confidence=.9,bbox_xyxy=[20,20,50,50])
    b=detection(21); b.update(class_name='truck',confidence=.6,bbox_xyxy=[21,20,51,50])
    c=detection(50); c.update(confidence=.8,bbox_xyxy=[50,20,80,50])
    kept,rejected,indices=deduplicate_observations([b,c,a])
    assert indices==[1,2] and kept==[c,a]
    assert len(rejected)==1 and rejected[0]['candidate_index']==0
    assert rejected[0]['kept_candidate_index']==2
    assert rejected[0]['reason']=='duplicate_instance_mask'


def test_confirmation_requires_consecutive_hits_and_tentatives_expire():
    tracker=FlowMaskTracker(max_lost=5,min_hits=3,tentative_max_lost=1)
    gray=np.zeros((120,140),np.uint8)
    for i in range(3):
        tracks,_=tracker.update([detection(20+i)],None,gray,.1)
        assert tracks[0].confirmed==(i==2)
    for i in range(5):
        tracks,retired=tracker.update([],None,gray,.1)
        assert tracks[0].id==1 and not retired
    tracks,retired=tracker.update([],None,gray,.1)
    assert retired==[1] and not tracks
    tracks,_=tracker.update([detection(20)],None,gray,.1)
    assert tracks[0].id==2 and not tracks[0].confirmed
    tracker.update([],None,gray,.1)
    tracks,retired=tracker.update([],None,gray,.1)
    assert retired==[2] and not tracks


def test_appearance_reassociation_rejects_wrong_color():
    import cv2
    gray=np.zeros((120,180),np.uint8)
    def picture(x,color):
        img=np.zeros((120,180,3),np.uint8);cv2.rectangle(img,(x,20),(x+30,50),color,-1)
        return img
    for color,expected_id in [((0,0,220),1),((220,0,0),2)]:
        tracker=FlowMaskTracker(max_lost=5)
        tracker.update([detection(20)],None,gray,.1,current_bgr=picture(20,(0,0,220)))
        tracker.update([],None,gray,.1)
        # IoU .15 fails old overlap threshold; gated same-color candidate recovers.
        tracks,_=tracker.update([detection(42)],None,gray,.1,current_bgr=picture(42,color))
        observed=[t for t in tracks if not t.lost]
        assert len(observed)==1 and observed[0].id==expected_id
        if expected_id==1: assert observed[0].reacquired_after_frames==1
