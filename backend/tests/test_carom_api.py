import json
from fastapi.testclient import TestClient
from app.main import app
from app import carom_api


def test_method_api_is_separate_and_reports_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(carom_api,'OUTPUT',tmp_path)
    client=TestClient(app)
    assert client.get('/api/carom-method/audit').status_code==503
    (tmp_path/'audit.json').write_text(json.dumps({'calibration':{'status':'experimental_unverified'}}))
    (tmp_path/'frames.jsonl').write_text(json.dumps({'frame_index':0,'objects':[{'id':1,'ground_xy':None}]})+'\n')
    assert client.get('/api/carom-method/audit').json()['calibration']['status']=='experimental_unverified'
    assert client.get('/api/carom-method/frames/0').json()['objects'][0]['ground_xy'] is None
    assert client.get('/api/carom-method/frames/-1').status_code==422
    assert client.get('/api/carom-method/frames/5').status_code==404


def test_method_reports_expose_unverified_calibration(tmp_path,monkeypatch):
    monkeypatch.setattr(carom_api,'OUTPUT',tmp_path)
    client=TestClient(app)
    for route in ('smoothing-report','calibration-readiness'):
        assert client.get('/api/carom-method/'+route).status_code==503
        (tmp_path/(route+'.json')).write_text(json.dumps({'accuracy_validated':False}))
        assert client.get('/api/carom-method/'+route).json()['accuracy_validated'] is False


def test_default_artifact_root_is_repository_not_backend(monkeypatch):
    from pathlib import Path
    expected=Path(carom_api.__file__).resolve().parents[2]/'outputs/carom-offline-v5-final'
    # Configuration override remains supported; test the repository layout itself.
    assert expected.parent.parent.joinpath('scripts/run_carom_method.py').is_file()


def test_run_info_never_advertises_incomplete_or_legacy_run(tmp_path,monkeypatch):
    monkeypatch.setattr(carom_api,'OUTPUT',tmp_path)
    client=TestClient(app)
    assert client.get('/api/carom-method/run-info').status_code==503
    for name in ('frames.jsonl','validation.png','mobile.mp4','bev-detail.mp4','verification.json',
                 'smoothing-report.json','calibration-readiness.json'):
        (tmp_path/name).write_text('{}')
    (tmp_path/'audit.json').write_text(json.dumps({'release_version':'offline-v4','frames':240,
        'source_fps':30.,'preview_frame':108,'source_sha256':'test','calibration':{'status':'experimental_unverified'}}))
    r=client.get('/api/carom-method/run-info').json()
    assert r['version']=='offline-v4' and r['api_namespace']=='/api/carom-method'
    assert r['accuracy_validated'] is False and r['frames']==240
