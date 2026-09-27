"""Generate an eight-lane baseline; export RGB inputs separately from oracle data.

Only this script's actors are removed. Existing vehicles/sensors prevent map replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import random
import shutil
import time
import traceback

import carla
import cv2
import numpy as np

from carom_sim.geometry import calibration, make_road, project, to_road

ROOT=Path(__file__).resolve().parent
EDGES=[(0,1),(1,3),(3,2),(2,0),(0,4),(4,5),(5,1),(5,7),(7,6),(6,4),(6,2),(7,3)]

def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def run(args):
    client=carla.Client(args.host,args.port);client.set_timeout(90)
    world=client.get_world()
    client_version=client.get_client_version();server_version=client.get_server_version()
    if client.get_client_version()!=client.get_server_version():raise RuntimeError('CARLA version mismatch')
    occupants=list(world.get_actors().filter('vehicle.*'))+list(world.get_actors().filter('sensor.*'))
    if occupants:raise RuntimeError('Existing vehicles/sensors found; preserve them. Clean this scenario explicitly before a new map.')
    out=ROOT/'runs'/time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('input','truth','diagnostics'): (out/name).mkdir()
    (out/'implementation').mkdir()
    shutil.copy2(Path(__file__),out/'implementation'/'capture_baseline.py')
    shutil.copy2(ROOT/'carom_sim'/'geometry.py',out/'implementation'/'geometry.py')
    write_json(out/'arguments.json',vars(args))
    (out/'truth'/'masks').mkdir()
    write_json(out/'previous_world.json',{'map':world.get_map().name,'settings':str(world.get_settings())})
    xodr=make_road();(out/'input'/'road.xodr').write_text(xodr,encoding='utf-8')
    print('Generating eight-lane OpenDRIVE world',flush=True)
    world=client.generate_opendrive_world(xodr,carla.OpendriveGenerationParameters(vertex_distance=2.,max_road_length=50.,wall_height=.1,additional_width=0.,smooth_junctions=False,enable_mesh_visibility=True))
    original_settings=world.get_settings();settings=world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=1/args.fps
    settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters(cloudiness=10,precipitation=0,wetness=0,fog_density=0,fog_distance=0,sun_altitude_angle=60,sun_azimuth_angle=130))
    # Standalone OpenDRIVE mesh lacks painted lane textures in this packaged build.
    # Draw static surveyed road markings IN the simulator, never on exported RGB.
    for lateral in [-14,-10.5,-7,-3.5,0,3.5,7,10.5,14]:
        solid=abs(lateral)==14 or lateral==0
        color=carla.Color(100,80,5) if lateral==0 else carla.Color(100,100,100)
        for start in range(0,500,12 if not solid else 500):
            world.debug.draw_line(carla.Location(x=start,y=lateral,z=.035),carla.Location(x=min(start+(4 if not solid else 500),500),y=lateral,z=.035),thickness=.12,color=color,life_time=0,persistent_lines=True)
    tm=client.get_trafficmanager(8050);tm.set_synchronous_mode(True);tm.set_random_device_seed(args.seed)
    tm.global_percentage_speed_difference(25);tm.set_global_distance_to_leading_vehicle(8)
    actors=[];sensors=[];writer=None;gt_writer=None;truth_file=None;frame_file=None;completed=False
    rng=random.Random(args.seed)
    try:
        road=world.get_map();library=world.get_blueprint_library()
        vehicle_ids=['vehicle.tesla.model3','vehicle.audi.a2','vehicle.lincoln.mkz_2020','vehicle.mercedes.coupe_2020']
        options=[b for b in library.filter('vehicle.*') if b.id in vehicle_ids]
        if not options:raise RuntimeError('No baseline car blueprints available')
        lane_records=[]
        for lane in [-4,-3,-2,-1,1,2,3,4]:
            center=road.get_waypoint_xodr(1,lane,150)
            if center is None:raise RuntimeError(f'Missing lane {lane}')
            lane_records.append({'road_id':1,'lane_id':lane,'width_m':center.lane_width,
                                 'center_lateral_m':center.transform.location.y,
                                 'forward_road_xy':[center.transform.get_forward_vector().y,center.transform.get_forward_vector().x]})
            for j,s in enumerate((130,172,214)):
                wp=road.get_waypoint_xodr(1,lane,s+abs(lane)*2)
                tf=wp.transform;tf.location.z+=.5
                bp=rng.choice(options);bp.set_attribute('role_name','carom_baseline')
                if bp.has_attribute('color'):bp.set_attribute('color',rng.choice(bp.get_attribute('color').recommended_values))
                actor=world.try_spawn_actor(bp,tf)
                if actor is None:raise RuntimeError(f'Spawn failed: lane={lane} s={s}')
                actors.append(actor);actor.set_autopilot(True,8050);tm.auto_lane_change(actor,False)
                tm.vehicle_percentage_speed_difference(actor,20+5*j)
        write_json(out/'input'/'lanes.json',{'coordinate_frame':'road_X=UE_Y; road_Y=UE_X-100; road_Z=UE_Z','lanes':lane_records,'longitudinal_roi_m':[0,140]})
        cam_tf=carla.Transform(carla.Location(x=90,y=-18,z=16),carla.Rotation(pitch=-22,yaw=8))
        world.get_spectator().set_transform(cam_tf)
        for _ in range(60):world.tick()
        queues=[]
        for kind in ('rgb','instance_segmentation'):
            bp=library.find('sensor.camera.'+kind)
            for key,value in {'image_size_x':str(args.width),'image_size_y':str(args.height),'fov':'90','sensor_tick':'0.0','lens_k':'0','lens_kcube':'0'}.items():
                if bp.has_attribute(key):bp.set_attribute(key,value)
            if kind=='rgb':
                for key,value in {'enable_postprocess_effects':'true','motion_blur_intensity':'0','bloom_intensity':'0','lens_flare_intensity':'0','blur_amount':'0','gamma':'2.2'}.items():
                    if bp.has_attribute(key):bp.set_attribute(key,value)
            sensor=world.spawn_actor(bp,cam_tf);sensors.append(sensor)
            q=queue.Queue();sensor.listen(q.put);queues.append(q)
        world.tick()  # Newly spawned sensors need a completed frame before querying transforms.
        print('Sensors ready; exporting calibration',flush=True)
        cal=calibration(sensors[0].get_transform().get_inverse_matrix(),args.width,args.height,90)
        cal['camera_transform_carla']={'location':[90,-18,16],'rotation_pitch_yaw_roll':[-22,8,0]}
        cal['sensor_attributes']=dict(sensors[0].attributes)
        write_json(out/'input'/'calibration.json',cal)
        writer=cv2.VideoWriter(str(out/'input'/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(args.width,args.height))
        gt_writer=cv2.VideoWriter(str(out/'diagnostics'/'oracle_boxes.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(args.width,args.height))
        if not writer.isOpened() or not gt_writer.isOpened():raise RuntimeError('Video encoder failed')
        truth_file=(out/'truth'/'actors.jsonl').open('w',encoding='utf-8')
        frame_file=(out/'input'/'frames.jsonl').open('w',encoding='utf-8')
        for index in range(args.frames):
            frame=world.tick();snapshot=world.get_snapshot()
            if snapshot.frame!=frame:raise RuntimeError('Another client advanced the synchronous world')
            images=[]
            for q in queues:
                while True:
                    im=q.get(timeout=30)
                    if im.frame==frame:images.append(im);break
                    if im.frame>frame:raise RuntimeError('Sensor frame mismatch')
            image=np.frombuffer(images[0].raw_data,np.uint8).reshape(args.height,args.width,4)[:,:,:3].copy()
            writer.write(image);overlay=image.copy()
            # Raw instance pixels are evaluation-only. No oracle mask reaches detector input.
            images[1].save_to_disk(str(out/'truth'/'masks'/f'{index:06d}.png'))
            records=[]
            for actor in actors:
                state=snapshot.find(actor.id);tf=state.get_transform();vel=state.get_velocity()
                vertices=to_road([[v.x,v.y,v.z] for v in actor.bounding_box.get_world_vertices(tf)])
                center=vertices.mean(axis=0);extent=actor.bounding_box.extent
                wp=road.get_waypoint(tf.location,project_to_road=False,lane_type=carla.LaneType.Driving)
                try:
                    uv=project(vertices,cal['projection_matrix'])
                    intersects=bool(uv[:,0].max()>=0 and uv[:,0].min()<args.width and uv[:,1].max()>=0 and uv[:,1].min()<args.height)
                except ValueError:uv=None;intersects=False
                record={'actor_id':actor.id,'blueprint':actor.type_id,'bbox_center_road_xyz':center.tolist(),
                        'ground_center_road_xy':center[:2].tolist(),'vertices_road_xyz':vertices.tolist(),
                        'dimensions_lwh':[2*extent.x,2*extent.y,2*extent.z],
                        'velocity_road_xy':[vel.y,vel.x],'heading_rad':float(np.arctan2(tf.get_forward_vector().x,tf.get_forward_vector().y)),
                        'lane_id_by_actor_origin':wp.lane_id if wp else None,
                        'projected_cuboid_px':uv.tolist() if uv is not None else None,
                        'in_camera_frustum':intersects,'visibility_note':'Frustum test only; not an occlusion visibility test'}
                records.append(record)
                if intersects:
                    for a,b in EDGES:
                        p1=tuple(np.clip(uv[a],-10000,10000).astype(int));p2=tuple(np.clip(uv[b],-10000,10000).astype(int))
                        cv2.line(overlay,p1,p2,(60,210,120),1,cv2.LINE_AA)
                    label=tuple(np.maximum(0,np.minimum([args.width-120,args.height-15],uv.mean(axis=0))).astype(int))
                    cv2.putText(overlay,f'GT {actor.id} / L{record["lane_id_by_actor_origin"]}',label,0,.38,(50,245,160),1,cv2.LINE_AA)
            cv2.rectangle(overlay,(0,0),(args.width,28),(15,20,24),-1)
            cv2.putText(overlay,'SIMULATOR GROUND TRUTH - NOT DETECTOR OUTPUT',(10,19),0,.45,(70,220,245),1,cv2.LINE_AA)
            gt_writer.write(overlay)
            meta={'index':index,'carla_frame':frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,'video_time_s':index/args.fps,'sensor_frames':[im.frame for im in images]}
            frame_file.write(json.dumps(meta)+'\n');truth_file.write(json.dumps({**meta,'objects':records},allow_nan=False)+'\n')
            if index in (0,args.frames//2):
                cv2.imwrite(str(out/'diagnostics'/f'rgb-{index:06d}.png'),image)
                cv2.imwrite(str(out/'diagnostics'/f'oracle-{index:06d}.png'),overlay)
            if index%30==0:print(f'Captured {index+1}/{args.frames}, frame={frame}, actors={len(records)}',flush=True)
        completed=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        traceback.print_exc()
        raise
    finally:
        for s in sensors:
            s.stop();s.destroy()
        if writer:writer.release()
        if gt_writer:gt_writer.release()
        if truth_file:truth_file.close()
        if frame_file:frame_file.close()
        for actor in actors:
            if completed and args.keep_scene:
                actor.set_autopilot(False,8050);actor.set_simulate_physics(False)
            else:actor.destroy()
        write_json(out/'run.json',{'completed':completed,'frames_requested':args.frames,'fps':args.fps,'seed':args.seed,
            'client_version':client_version,'server_version':server_version,
            'kept_actor_ids':[a.id for a in actors] if completed and args.keep_scene else [],
            'scene_after_capture':'frozen owned actors for inspection' if completed and args.keep_scene else 'owned actors removed',
            'protocol':'RGB + static calibration/map are inputs; truth/ and diagnostics/ are evaluator-only',
            'traffic':'Traffic Manager physical simulation; lane changes disabled baseline',
            'real_world_accuracy_validated':False})
        if completed:
            write_json(out/'input'/'manifest.json',{'video':'rgb.mp4','frames':args.frames,'fps':args.fps,
                       'sha256':hashlib.sha256((out/'input'/'rgb.mp4').read_bytes()).hexdigest(),
                       'calibration':'calibration.json','lanes':'lanes.json','frame_index':'frames.jsonl'})
            print('RUN='+str(out),flush=True)
        tm.set_synchronous_mode(False);world.apply_settings(original_settings)
    write_json(out/'cleanup.json',{'settings_restored':True,'traffic_manager_synchronous':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2000)
    p.add_argument('--frames',type=int,default=240);p.add_argument('--fps',type=int,default=20)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--keep-scene',action='store_true')
    a=p.parse_args()
    if not 1<=a.frames<=12000 or not 10<=a.fps<=60: p.error('frames 1..12000; fps 10..60')
    run(a)
