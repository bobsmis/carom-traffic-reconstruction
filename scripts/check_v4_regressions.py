"""Integration regressions for the supplied 240-frame user-bridge fixture only."""
import argparse
import hashlib
import json
from pathlib import Path


def check(output):
    audit=json.loads((output/'audit.json').read_text())
    assert audit['source_sha256']=='f6106e513f8f454097c1d7160ae495bb732597e110afaa6f39034358b2cfd4a8', 'wrong regression fixture'
    rows=[json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]
    source=Path(audit['offline_smoothing']['input_path'])/'frames.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['offline_smoothing']['input_sha256']
    online=[json.loads(s) for s in source.read_text().splitlines()]
    assert len(rows)==len(online)==240
    assert [[r['id'] for r in f['objects']] for f in rows]==[[r['id'] for r in f['objects']] for f in online]
    id8=[]
    for index in (57,58,59,60):
        r=next(r for r in rows[index]['objects'] if r['id']==8)
        assert r['observed'] and r['lost_frames']==0 and r['ground_xy'] is not None
        assert r['geometry_visibility']['publish']
        id8.append({'frame':index,'ground_xy':r['ground_xy'],'geometry_status':r['geometry_status'],
                    'measurement_status':r.get('measurement_status')})
    rejected=[]
    for f in rows:
        for r in f['objects']:
            e=r.get('offline_estimate')
            if not e: continue
            assert r['geometry_status']=='smoothed'
            assert (r['measurement_status']=='accepted')==e['measurement_used']
            if r['online_estimate']['geometry_status']=='measured' and not e['measurement_used']:
                assert r['measurement_status']=='rejected'
                rejected.append({'frame':f['frame_index'],'id':r['id']})
    assert rejected, 'fixture should exercise rejection labels'
    report={'passed':True,'scope':'user-bridge 240-frame fixture; not positional accuracy',
            'id8_gap_regression':id8,'rejected_measurements_correctly_labelled':rejected,
            'same_frame_id_sets':True,'input_hash_verified':True}
    (output/'regressions.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    check(parser.parse_args().output)
