"""CARLA left-handed world -> right-handed CAROM road frame -> CV camera."""
import numpy as np

def calibration(inverse_camera, width, height, horizontal_fov, origin_x=100.):
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov) / 2))
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    # Road X = UE Y (lateral); road Y = UE X - origin; road Z = UE Z.
    road_to_ue = np.array([[0,1,0,origin_x],[1,0,0,0],[0,0,1,0],[0,0,0,1.]])
    ue_camera_to_cv = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1.]])
    extrinsic = ue_camera_to_cv @ np.asarray(inverse_camera) @ road_to_ue
    P = K @ extrinsic[:3]
    return {'intrinsic_matrix':K.tolist(),'world_to_camera_cv':extrinsic.tolist(),
            'projection_matrix':P.tolist(),'image_to_ground':np.linalg.inv(P[:,[0,1,3]]).tolist(),
            'road_to_carla_world':road_to_ue.tolist(),'distortion':[0,0,0,0,0],
            'width':width,'height':height,'horizontal_fov_deg':horizontal_fov,
            'provenance':'CARLA simulator configuration, not surveyed real-world calibration'}

def project(points, projection):
    q = np.c_[np.asarray(points),np.ones(len(points))] @ np.asarray(projection).T
    if np.any(q[:,2] <= 0): raise ValueError('projection includes points behind camera')
    return q[:,:2]/q[:,2:]

def to_road(points, origin_x=100.):
    p=np.asarray(points,dtype=float)
    return np.stack([p[...,1],p[...,0]-origin_x,p[...,2]],axis=-1)

def make_road(length=500):
    if not 100 <= length <= 5000:raise ValueError('road length outside supported range')
    def side(sign):
        return ''.join(f'<lane id="{sign*i}" type="driving" level="false"><link/><width sOffset="0" a="3.5" b="0" c="0" d="0"/><roadMark sOffset="0" type="{"solid" if i==4 else "broken"}" weight="standard" color="white" width="0.15" laneChange="both"/><speed sOffset="0" max="13.9" unit="m/s"/></lane>' for i in range(1,5))
    return f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE><header revMajor="1" revMinor="4" name="CAROM_EightLane_Baseline" version="1.0" date="2026-09-26" north="20" south="-20" east="{length}" west="0"/>
<road name="CAROM straight road" length="{length}" id="1" junction="-1"><link/><type s="0" type="town"><speed max="13.9" unit="m/s"/></type>
<planView><geometry s="0" x="0" y="0" hdg="0" length="{length}"><line/></geometry></planView><elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile><lateralProfile/>
<lanes><laneSection s="0"><left>{side(1)}</left><center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid solid" weight="standard" color="yellow" width="0.15"/></lane></center><right>{side(-1)}</right></laneSection></lanes><objects/><signals/></road></OpenDRIVE>'''
