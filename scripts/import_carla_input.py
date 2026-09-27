"""Input-only bridge. This importer never opens the simulator truth directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def convert(source, name):
    if not name.replace('-','').replace('_','').isalnum(): raise ValueError('invalid scene name')
    manifest=json.loads((source/'manifest.json').read_text())
    if hashlib.sha256((source/'rgb.mp4').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('video hash mismatch')
    output=ROOT/'configs'/f'{name}.json'
    if output.exists():raise FileExistsError(output)
    lane_model=json.loads((source/'lanes.json').read_text())
    lanes=lane_model['lanes']
    lower=min(l['center_lateral_m']-l['width_m']/2 for l in lanes)
    upper=max(l['center_lateral_m']+l['width_m']/2 for l in lanes)
    near,far=lane_model['longitudinal_roi_m']
    destination=ROOT/'data'/'carla'/name
    destination.mkdir(parents=True,exist_ok=False)
    for filename in ('rgb.mp4','calibration.json','lanes.json','road.xodr','frames.jsonl','manifest.json'):
        shutil.copy2(source/filename,destination/filename)
    camera=json.loads((source/'calibration.json').read_text())
    # Static road model and detector settings; no actor truth or masks are imported.
    config=json.loads((ROOT/'configs/user_bridge_traffic.json').read_text())
    config.update(scene_id=name,dataset='CARLA fixed infrastructure camera baseline',camera_id='carla_fixed_rgb',
                  source_url='local-simulation://'+name,source_sha256=manifest['sha256'],
                  video_path=str((destination/'rgb.mp4').relative_to(ROOT)),
                  crop={'x':0,'y':0,'width':camera['width'],'height':camera['height']})
    config['planar_calibration']={'method':'carla_exact_extrinsics','intrinsic_matrix':camera['intrinsic_matrix'],
        'world_to_camera_cv':camera['world_to_camera_cv'],'distortion':camera['distortion'],
        'provenance':camera['provenance']}
    config['measurement_region'].update(ground_polygon_m=[[lower,near],[upper,near],[upper,far],[lower,far]],
        min_bbox_width_px=12,min_bbox_height_px=12,min_mask_area_px=90,max_projection_sigma_m=2.5,
        max_camera_ground_distance_m=180,note='Flat simulated road; bounds from authored map. Pixel gates remain algorithmic.')
    config['processing'].update(max_frames=manifest['frames'],output_fps=manifest['fps'])
    config['road_map']={'lane_width_m':lanes[0]['width_m'],'visible_lane_count':len(lanes),'source':'authored OpenDRIVE',
        'lanes':lanes,'longitudinal_bounds_m':[near,far],
        'lane_mapping_note':'Signed OpenDRIVE lane IDs from static map; no vehicle state or position snapping.'}
    config['provenance']={'llm_runtime_calls':0,'manual_vehicle_state_fields':0,
        'input_files_read':['rgb.mp4','calibration.json','lanes.json','road.xodr','frames.jsonl','manifest.json'],
        'vehicle_annotations':'local YOLO only; simulator actor states and masks are evaluator-only'}
    output.write_text(json.dumps(config,indent=2))
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input_directory',type=Path);p.add_argument('--name',required=True)
    a=p.parse_args();convert(a.input_directory,a.name)
