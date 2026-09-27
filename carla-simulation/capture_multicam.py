"""Synchronous multicamera acquisition. Actor truth never enters input/."""
import argparse
import hashlib
import json
import queue
import random
import shutil
import subprocess
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from carom_sim.geometry import calibration, make_road, to_road, project
from carom_sim.long_tunnel import START,LENGTH,ROAD_LENGTH,ROLE,build,camera_nodes

ROOT=Path(__file__).resolve().parent
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def run(args):
    client=carla.Client('127.0.0.1',2000);client.set_timeout(120)
    w=client.get_world()
    if list(w.get_actors().filter('vehicle.*'))+list(w.get_actors().filter('sensor.*')):raise RuntimeError('World still contains vehicles/sensors')
    out=ROOT/'runs'/time.strftime('%Y%m%d-%H%M%S');out.mkdir()
    for d in ('input','truth','diagnostics','implementation'):(out/d).mkdir()
    for p in [Path(__file__),ROOT/'carom_sim/long_tunnel.py',ROOT/'carom_sim/geometry.py']:
        shutil.copy2(p,out/'implementation'/p.name)
    save(out/'arguments.json',vars(args));nodes=camera_nodes()
    registry={'schema':'multicam-v1','coordinate_frame':'X=UE_Y,Y=UE_X-56,Z=UE_Z',
              'tunnel_length_m':LENGTH,'tunnel_start_x':START,'cameras':nodes,
              'coverage_note':'Authored ground ROI overlap; actual visibility/pixel resolution may reduce valid coverage.'}
    save(out/'input/cameras.json',registry)
    xodr=make_road(ROAD_LENGTH)
    w=client.generate_opendrive_world(xodr,carla.OpendriveGenerationParameters(vertex_distance=2.,max_road_length=50.,wall_height=.1,additional_width=0.,smooth_junctions=False))
    original=w.get_settings();settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=1/args.fps
    settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10;w.apply_settings(settings)
    tm=client.get_trafficmanager(8050);tm.set_synchronous_mode(True);tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(8)
    vehicles=[];props=[];sensors=[];writers=[];handles=[];cams=[];boxes={};collision_sensors={};event_queue=queue.Queue()
    rng=random.Random(args.seed);completed=False;count=0
    try:
        print('Building 1240 m semicircular tunnel',flush=True);props=build(w)
        road=w.get_map();lib=w.get_blueprint_library()
        for lateral in (-14,-10.5,-7,-3.5,0,3.5,7,10.5,14):
            for s in range(0,ROAD_LENGTH,12):
                w.debug.draw_line(carla.Location(s,lateral,.035),carla.Location(min(s+(12 if lateral in (-14,0,14) else 4),ROAD_LENGTH),lateral,.035),
                    thickness=.12,color=carla.Color(100,80,5) if lateral==0 else carla.Color(120,120,120),life_time=0,persistent_lines=True)
        options=[lib.find(i) for i in ['vehicle.tesla.model3','vehicle.audi.a2','vehicle.mercedes.sprinter','vehicle.carlamotors.carlacola','vehicle.mitsubishi.fusorosa']]
        def spawn(lane,s):
            tf=road.get_waypoint_xodr(1,lane,s).transform;tf.location.z+=.5
            if any(a.is_alive and abs(a.get_location().y-tf.location.y)<3.3 and abs(a.get_location().x-tf.location.x)<24 for a in vehicles):return None
            bp=rng.choice(options);bp.set_attribute('role_name',ROLE)
            a=w.try_spawn_actor(bp,tf)
            if a:
                vehicles.append(a);boxes[a.id]=a.bounding_box;a.set_autopilot(True,8050)
                tm.auto_lane_change(a,args.lane_changes);tm.vehicle_percentage_speed_difference(a,rng.uniform(5,25))
                a.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position|carla.VehicleLightState.LowBeam))
                sensor=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=a)
                collision_sensors[a.id]=sensor
                sensor.listen(lambda e:event_queue.put({'type':'collision_sensor','carla_frame':e.frame,'actor_id':e.actor.id,'other_actor_id':e.other_actor.id,'impulse':e.normal_impulse.length()}));sensors.append(sensor)
            return a
        # Start empty. All demand enters outside the displayed tunnel portals;
        # never instantiate a vehicle at a camera's interior checkpoint.
        entry_stations={lane:START-20 if lane<0 else START+LENGTH+20 for lane in (-4,-3,-2,-1,1,2,3,4)}
        save(out/'diagnostics/spawn_policy.json',{'initial_vehicles':0,'entry_stations_ue_x':entry_stations,
             'display_boundary_ue_x':[START,START+LENGTH],'buffer_m':20})
        brightness=args.brightness
        def lighting(value):
            # Physical CARLA weather/ambient setting, not a CSS/post-export brightening.
            w.set_weather(carla.WeatherParameters(sun_altitude_angle=5+value*.75,sun_azimuth_angle=130,
                cloudiness=100-value,precipitation=0,wetness=0,fog_density=0))
        lighting(brightness)
        for _ in range(30):w.tick()
        for node in nodes:
            folder=out/'input'/node['id'];folder.mkdir();truthdir=out/'truth'/node['id'];truthdir.mkdir();(truthdir/'masks').mkdir()
            (folder/'road.xodr').write_text(xodr)
            lane_data=[]
            for lane in (-4,-3,-2,-1,1,2,3,4):
                wp=road.get_waypoint_xodr(1,lane,node['origin_x']+50)
                lane_data.append({'road_id':1,'lane_id':lane,'width_m':wp.lane_width,'center_lateral_m':wp.transform.location.y,
                    'forward_road_xy':[wp.transform.get_forward_vector().y,wp.transform.get_forward_vector().x]})
            save(folder/'lanes.json',{'lanes':lane_data,'longitudinal_roi_m':[0,150]})
            tf=carla.Transform(carla.Location(*node['location']),carla.Rotation(pitch=node['pitch'],yaw=node['yaw']))
            if not cams:w.get_spectator().set_transform(tf)
            queues=[];local_sensors=[]
            for kind in ('rgb','instance_segmentation'):
                bp=lib.find('sensor.camera.'+kind)
                attributes={'image_size_x':str(args.width),'image_size_y':str(args.height),'fov':'90','sensor_tick':'0',
                    'lens_k':'0','lens_kcube':'0','motion_blur_intensity':'0','bloom_intensity':'0','lens_flare_intensity':'0'}
                for key,value in attributes.items():
                    if bp.has_attribute(key):bp.set_attribute(key,value)
                sensor=w.spawn_actor(bp,tf);q=queue.Queue();sensor.listen(q.put);sensors.append(sensor);local_sensors.append(sensor);queues.append(q)
            writer=cv2.VideoWriter(str(folder/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(args.width,args.height))
            if not writer.isOpened():raise RuntimeError('video encoder failed')
            writers.append(writer);frame_file=(folder/'frames.jsonl').open('w');truth_file=(truthdir/'actors.jsonl').open('w');handles.extend([frame_file,truth_file])
            cams.append({'node':node,'folder':folder,'truthdir':truthdir,'sensors':local_sensors,'queues':queues,'writer':writer,'frames':frame_file,'truth':truth_file})
        w.tick()
        for cam in cams:
            cam['cal']=calibration(cam['sensors'][0].get_transform().get_inverse_matrix(),args.width,args.height,90,cam['node']['origin_x'])
            save(cam['folder']/'calibration.json',cam['cal'])
        print('Ten camera pairs ready',flush=True)
        for index in range(args.frames):
            if index and (ROOT/'stop.request').exists():break
            vehicles=[a for a in vehicles if a.is_alive]
            for a in list(vehicles):
                x=a.get_location().x
                if x<START-35 or x>START+LENGTH+35:
                    event_queue.put({'type':'exit_remove','index':index,'actor_id':a.id,'ue_x':x})
                    sensor=collision_sensors.pop(a.id,None)
                    if sensor is not None and sensor.is_alive:sensor.stop();sensor.destroy()
                    a.destroy();vehicles.remove(a)
                    boxes.pop(a.id,None)
            if index%args.fps==0:
                controls=ROOT/'scene-controls.json'
                if controls.exists():
                    request=json.loads(controls.read_text());value=float(request.get('brightness',brightness))
                    if value!=brightness:brightness=value;lighting(value);event_queue.put({'type':'ambient_brightness_changed','index':index,'value':value})
                if len(vehicles)<100:
                    lanes=list(entry_stations);rng.shuffle(lanes)
                    for lane in lanes:
                        if len(vehicles)>=100:break
                        if rng.random()>.65:continue
                        station=entry_stations[lane];a=spawn(lane,station)
                        if a:event_queue.put({'type':'entry_spawn','index':index,'actor_id':a.id,'lane':lane,'ue_x':station})
            if args.lane_changes and vehicles and index and index%(args.fps*4)==0:
                a=rng.choice(vehicles);tm.force_lane_change(a,rng.choice([True,False]));event_queue.put({'type':'lane_change_requested','index':index,'actor_id':a.id})
            frame=w.tick();snap=w.get_snapshot()
            if snap.frame!=frame:raise RuntimeError('Multiple synchronous tick clients detected')
            for cam in cams:
                images=[]
                for q in cam['queues']:
                    while True:
                        im=q.get(timeout=60)
                        if im.frame==frame:images.append(im);break
                        if im.frame>frame:raise RuntimeError('sensor synchronization mismatch')
                image=np.frombuffer(images[0].raw_data,np.uint8).reshape(args.height,args.width,4)[:,:,:3].copy();cam['writer'].write(image)
                if index%2==0:
                    p=out/'diagnostics'/(cam['node']['id']+'.jpg');tmp=p.with_suffix('.tmp.jpg');cv2.imwrite(str(tmp),image);tmp.replace(p)
                images[1].save_to_disk(str(cam['truthdir']/'masks'/f'{index:06}.png'))
                meta={'index':index,'carla_frame':frame,'timestamp_s':snap.timestamp.elapsed_seconds,'video_time_s':index/args.fps,'sensor_frames':[im.frame for im in images]}
                cam['frames'].write(json.dumps(meta)+'\n');objects=[]
                for a in vehicles:
                    state=snap.find(a.id)
                    if state is None:continue
                    tf=state.get_transform();bb=boxes[a.id];vertices=to_road([[v.x,v.y,v.z] for v in bb.get_world_vertices(tf)],cam['node']['origin_x'])
                    try:uv=project(vertices,cam['cal']['projection_matrix']);visible=bool(uv[:,0].max()>=0 and uv[:,0].min()<args.width and uv[:,1].max()>=0 and uv[:,1].min()<args.height)
                    except ValueError:uv=None;visible=False
                    objects.append({'actor_id':a.id,'blueprint':a.type_id,'ground_center_road_xy':vertices.mean(axis=0)[:2].tolist(),
                        'dimensions_lwh':[2*bb.extent.x,2*bb.extent.y,2*bb.extent.z],
                        'heading_rad':float(np.arctan2(tf.get_forward_vector().x,tf.get_forward_vector().y)),
                        'projected_cuboid_px':uv.tolist() if uv is not None else None,'in_camera_frustum':visible})
                cam['truth'].write(json.dumps({**meta,'objects':objects})+'\n')
            with (out/'truth/events.jsonl').open('a') as f:
                while not event_queue.empty():f.write(json.dumps(event_queue.get())+'\n')
            count+=1
            if index%10==0:print(f'Captured {count}/{args.frames}, cameras=10, actors={len(vehicles)}',flush=True)
        # Publish browser-compatible RGB independently of the perception stage.
        # Close encoders before reading MP4 headers; atomic rename prevents partial playback.
        for writer in writers:writer.release()
        import imageio_ffmpeg
        for cam in cams:
            target=out/'web'/cam['node']['id'];target.mkdir(parents=True,exist_ok=True)
            temporary=target/'raw.encoding.mp4'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(cam['folder']/'rgb.mp4'),
                '-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
            temporary.replace(target/'raw.mp4')
        completed=True
    finally:
        for s in sensors:
            if s.is_alive:s.stop();s.destroy()
        for f in handles:f.close()
        for writer in writers:writer.release()
        for a in vehicles:
            if a.is_alive:
                if completed:a.set_autopilot(False,8050);a.set_simulate_physics(False)
                else:a.destroy()
        if not completed:
            for a in props:
                if a.is_alive:a.destroy()
        tm.set_synchronous_mode(False);w.apply_settings(original)
        save(out/'run.json',{'completed':completed,'frames_requested':count,'fps':args.fps,'schema':'multicam-v1','cameras':10,
            'kept_actor_ids':[a.id for a in vehicles+props if a.is_alive], 'tunnel_length_m':LENGTH})
        if completed:
            for cam in cams:
                save(cam['folder']/'manifest.json',{'video':'rgb.mp4','frames':count,'fps':args.fps,
                    'sha256':hashlib.sha256((cam['folder']/'rgb.mp4').read_bytes()).hexdigest()})
            save(out/'input/manifest.json',{'schema':'multicam-v1','frames':count,'fps':args.fps,'cameras':10})
            print('RUN='+str(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=1800);p.add_argument('--fps',type=int,default=10)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540)
    p.add_argument('--seed',type=int,default=83);p.add_argument('--brightness',type=float,default=70)
    p.add_argument('--lane-changes',action='store_true');a=p.parse_args()
    if not 1<=a.frames<=3600 or not 10<=a.fps<=20 or not 0<=a.brightness<=100:p.error('invalid capture settings')
    run(a)
