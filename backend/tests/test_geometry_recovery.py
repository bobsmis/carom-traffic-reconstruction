from app.pipeline.geometry_recovery import GeometryRecovery


def fit(y=50.,dims=None):
    return {'status':'fitted','silhouette_hull_iou':.9,'support_rmse_m':.05,'support_rank':5,
            'dimensions_lwh':dims or [4.5,1.8,1.5],'ground_xy':[2.,y]}


def test_recovery_needs_six_consistent_independent_observations():
    r=GeometryRecovery()
    for i in range(5):r.observe(fit(50+i/3),'car',i,.9)
    assert r.candidate(30) is None
    r.observe(fit(50+5/3),'car',5,.9)
    c=r.candidate(30)
    assert c['sample_count']==6 and abs(c['velocity_xy_mps'][1]-10)<1e-8


def test_class_switch_gap_and_weak_association_break_evidence():
    for mode in ('class','gap','association'):
        r=GeometryRecovery()
        for i in range(6):
            r.observe(fit(),'truck' if mode=='class' and i==5 else 'car',i+2 if mode=='gap' and i==5 else i,
                      .1 if mode=='association' and i==5 else .9)
        assert r.candidate(30) is None


def test_dimension_outlier_and_impossible_motion_rejected():
    for mode in ('size','motion'):
        r=GeometryRecovery()
        for i in range(6):r.observe(fit(i*10 if mode=='motion' else 50,[8,3,4] if mode=='size' and i==5 else None),'car',i,.9)
        assert r.candidate(30) is None


def test_initialization_requires_three_consistent_shapes_not_one_good_fit():
    r=GeometryRecovery()
    r.observe(fit(dims=[4.5,2.2,3.3]),'truck',0,.9)
    assert r.candidate(30,minimum=3) is None
    # A class change discards the mixed-mask bootstrap hypothesis.
    for i in range(1,4):r.observe(fit(60-i/3),'car',i,.9)
    c=r.candidate(30,minimum=3)
    assert c['dimensions_lwh']==[4.5,1.8,1.5] and c['sample_count']==3


def test_offline_smoothing_does_not_bridge_geometry_revision():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from smooth_carom_offline import segments
    rows=[{'frame_index':i,'objects':[{'id':1,'ground_xy':[0,i],'geometry_epoch':int(i>=3)}]} for i in range(6)]
    chunks=list(segments(rows))
    assert len(chunks)==2 and [len(c) for _,c in chunks]==[3,3]
