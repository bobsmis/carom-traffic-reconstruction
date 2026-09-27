"""SUMO/TraCI replay of RGB-derived estimates, never simulation truth.

SUMO is a visualization adapter here, not an independent accuracy reference.
keepRoute=2 intentionally preserves off-road estimates rather than snapping errors
onto lanes. The web renderer consumes positions read back from SUMO.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,'/usr/share/sumo/tools')
import traci
import cv2
from process_multicam import map_view,encode

def write_xml(path,root):
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def build_network(folder,length):
    folder.mkdir(parents=True,exist_ok=True)
    nodes=ET.Element('nodes')
    ET.SubElement(nodes,'node',id='west',x='0',y='0',type='dead_end')
    ET.SubElement(nodes,'node',id='east',x=str(length),y='0',type='dead_end')
    edges=ET.Element('edges')
    for eid,a,b in [('eastbound','west','east'),('westbound','east','west')]:
        ET.SubElement(edges,'edge',{'id':eid,'from':a,'to':b,'numLanes':'4','width':'3.5','speed':'22.22','spreadType':'right'})
    write_xml(folder/'road.nod.xml',nodes);write_xml(folder/'road.edg.xml',edges)
    subprocess.run(['netconvert','--node-files',str(folder/'road.nod.xml'),'--edge-files',str(folder/'road.edg.xml'),
                    '--offset.disable-normalization','true','--output-file',str(folder/'road.net.xml')],check=True)
    config=ET.Element('configuration');inp=ET.SubElement(config,'input')
    ET.SubElement(inp,'net-file',value='road.net.xml')
    write_xml(folder/'replay.sumocfg',config)

def run(web,rig,gui=False):
    data=json.loads((web/'global.json').read_text());registry=json.loads(rig.read_text())
    folder=web/'sumo';build_network(folder,registry['tunnel_length_m'])
    fps=float(data['fps']);nodes=registry['cameras'];writers={};rows=[];max_error=0.
    traci.start(['sumo-gui' if gui else 'sumo','-c',str(folder/'replay.sumocfg'),'--step-length',str(1/fps),
        '--no-step-log','true','--collision.action','none','--time-to-teleport','-1','--start','--quit-on-end'])
    active=set()
    try:
        traci.route.add('east',['eastbound']);traci.route.add('west',['westbound'])
        for node in nodes:
            cid=node['id'];p=folder/(cid+'.mp4')
            w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(960,540))
            if not w.isOpened():raise RuntimeError('SUMO video encoder failed')
            writers[cid]=(w,p)
        for frame in data['frames']:
            current={f'G{o["id"]}' for o in frame['objects']}
            for vid in active-current:
                if vid in traci.vehicle.getIDList():traci.vehicle.remove(vid)
            active.intersection_update(current)
            actual=[]
            for obj in frame['objects']:
                vid=f'G{obj["id"]}';heading=float(obj['heading']);length,width,height=map(float,obj['dimensions'])
                if not all(math.isfinite(v) for v in [*obj['xy'],heading,length,width,height]) or min(length,width,height)<=0:
                    raise ValueError('nonfinite or invalid input estimate')
                if vid not in active:
                    traci.vehicle.add(vid,'east' if math.sin(heading)>=0 else 'west',depart='now')
                    traci.vehicle.setSpeedMode(vid,0);traci.vehicle.setLaneChangeMode(vid,0);active.add(vid)
                traci.vehicle.setLength(vid,length);traci.vehicle.setWidth(vid,width);traci.vehicle.setHeight(vid,height)
                traci.vehicle.setSpeed(vid,0)
                # SUMO uses the front bumper midpoint, our algorithm uses cuboid center.
                x=float(obj['xy'][1])+length/2*math.sin(heading)
                y=float(obj['xy'][0])+length/2*math.cos(heading)
                traci.vehicle.moveToXY(vid,'',-1,x,y,angle=math.degrees(heading)%360,keepRoute=2,matchThreshold=2000)
            # Newly added vehicles are pending until SUMO advances; readback before
            # this step returns INVALID_DOUBLE_VALUE rather than a real position.
            traci.simulationStep()
            for obj in frame['objects']:
                vid=f'G{obj["id"]}';heading=float(obj['heading']);length=float(obj['dimensions'][0])
                sx,sy=traci.vehicle.getPosition(vid)
                out=copy.deepcopy(obj);out['xy']=[sy-length/2*math.cos(heading),sx-length/2*math.sin(heading)]
                error=math.dist(out['xy'],obj['xy']);max_error=max(max_error,error)
                if error>.01:raise RuntimeError(f'SUMO replay coordinate mismatch: {error:.3f} m')
                actual.append(out)
            rows.append({'frame_index':frame['frame_index'],'time_s':frame['time_s'],'objects':actual})
            for node in nodes:
                image=map_view([o for o in actual if node['id'] in o['cameras']],nodes,node['global_offset_m'])
                cv2.rectangle(image,(0,505),(960,540),(30,35,39),-1)
                cv2.putText(image,'SUMO / TraCI replay of vision estimates | no lane snapping | not ground truth',(20,530),0,.4,(190,215,230),1)
                writers[node['id']][0].write(image)
    finally:
        for w,_ in writers.values():w.release()
        traci.close()
    for cid,(_,p) in writers.items():encode(p,web/cid/'sumo.mp4')
    (folder/'replay.json').write_text(json.dumps({'frames':rows,'source':'rgb_algorithm_global.json','truth_used':False,
        'lane_snapping':False,'coordinate_error_max_m':max_error,'sumo_version':subprocess.check_output(['sumo','--version'],text=True).splitlines()[0]}))
    print('SUMO_REPLAY='+str(folder),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--web',type=Path,required=True);p.add_argument('--rig',type=Path,required=True)
    p.add_argument('--gui',action='store_true');a=p.parse_args();run(a.web,a.rig,a.gui)
