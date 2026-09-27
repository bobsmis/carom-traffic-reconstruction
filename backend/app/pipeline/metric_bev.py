"""Native-resolution BEV detail: one shared pixels-per-metre for both axes."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .carom_geometry import cuboid_vertices


@dataclass(frozen=True)
class MetricViewport:
    pixels_per_meter: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self):
        if (not np.isfinite([self.pixels_per_meter,self.x_min,self.x_max,self.y_min,self.y_max]).all()
                or not 1<=self.pixels_per_meter<=40 or self.x_min>=self.x_max or self.y_min>=self.y_max):
            raise ValueError('invalid metric display extent or scale')
        if self.width*self.height>16_000_000:
            raise ValueError('metric display exceeds raster budget')

    @property
    def width(self):
        return math.ceil((self.y_max-self.y_min)*self.pixels_per_meter/2)*2

    @property
    def height(self):
        return math.ceil((self.x_max-self.x_min)*self.pixels_per_meter/2)*2

    @property
    def transform(self):
        s=self.pixels_per_meter
        return np.array([[0.,-s,s*self.y_max],[s,0.,-s*self.x_min],[0.,0.,1.]])

    def project(self,xy):
        points=np.asarray(xy,float).reshape(-1,2)
        return (np.c_[points,np.ones(len(points))]@self.transform.T)[:,:2]

    def point(self,xy):
        return tuple(np.round(self.project([xy])[0]).astype(int))


class MetricBevDetail:
    HEADER=64
    FOOTER=60

    def __init__(self,camera,config,video,pixels_per_meter=20.):
        self.polygon=np.asarray(config['measurement_region']['ground_polygon_m'],float)
        low=self.polygon.min(axis=0); high=self.polygon.max(axis=0)
        self.viewport=MetricViewport(pixels_per_meter,low[0]-2,high[0]+2,low[1]-4,high[1]+4)
        v=self.viewport
        homography=v.transform@np.linalg.inv(camera.H)
        capture=cv2.VideoCapture(str(video)); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        samples=[]; crop=config['crop']
        for index in np.linspace(0,max(0,count-1),15,dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(index)); ok,frame=capture.read()
            if ok:
                frame=frame[crop['y']:crop['y']+crop['height'],crop['x']:crop['x']+crop['width']]
                samples.append(cv2.warpPerspective(frame,homography,(v.width,v.height)))
        capture.release()
        if not samples: raise ValueError('no readable background video frames')
        self.background=np.median(samples,axis=0).astype(np.uint8)
        mask=np.zeros((v.height,v.width),np.uint8)
        cv2.fillConvexPoly(mask,np.round(v.project(self.polygon)).astype(np.int32),255)
        self.background[mask==0]=(self.background[mask==0]*.28).astype(np.uint8)
        grid=self.background.copy()
        for x in range(math.ceil(v.x_min),math.floor(v.x_max)+1):
            cv2.line(grid,v.point([x,v.y_min]),v.point([x,v.y_max]),(160,180,170) if x%5==0 else (90,110,100),1)
        for y in range(math.ceil(v.y_min),math.floor(v.y_max)+1):
            cv2.line(grid,v.point([v.x_min,y]),v.point([v.x_max,y]),(160,180,170) if y%5==0 else (90,110,100),1)
        self.background=cv2.addWeighted(grid,.38,self.background,.62,0)
        # Lane guides remain nominal. They do not alter vehicle positions.
        width=config['road_map']['lane_width_m']; lanes=config['road_map']['visible_lane_count']
        for x in np.linspace(-lanes*width/2,lanes*width/2,lanes+1):
            cv2.line(self.background,v.point([x,low[1]]),v.point([x,high[1]]),(155,155,155),1,cv2.LINE_AA)
        cv2.polylines(self.background,[np.round(v.project(self.polygon)).astype(np.int32)],True,(90,245,160),2,cv2.LINE_AA)

    def render(self,record):
        v=self.viewport; plane=self.background.copy()
        for obj in record['objects']:
            if obj['ground_xy'] is None: continue
            xy=obj['ground_xy']; dims=obj['dimensions_lwh']; theta=obj['heading_rad']
            hue=np.uint8([[[obj['id']*37%180,190,255]]])
            color=tuple(int(c) for c in cv2.cvtColor(hue,cv2.COLOR_HSV2BGR)[0,0])
            corners=cuboid_vertices(xy,theta,dims)[:4,:2]
            points=np.round(v.project(corners)).astype(np.int32)
            painted=plane.copy(); cv2.fillConvexPoly(painted,points,color)
            # Constant opacity avoids brightness flicker on measurement/prediction switches.
            plane=cv2.addWeighted(painted,.55,plane,.45,0)
            cv2.polylines(plane,[points],True,color,2,cv2.LINE_AA)
            center=v.point(xy)
            front=np.asarray(xy)+np.array([math.cos(theta),math.sin(theta)])*dims[0]/2
            cv2.arrowedLine(plane,center,v.point(front),(250,250,250),2,tipLength=.35)
            label=f"#{obj['id']}"+(' S' if obj['geometry_status']=='smoothed' else ' P' if obj['geometry_status']=='predicted' else '')
            cv2.putText(plane,label,(center[0]+6,center[1]-7),0,.55,color,2,cv2.LINE_AA)
        canvas=np.full((v.height+self.HEADER+self.FOOTER,v.width,3),(18,22,25),np.uint8)
        canvas[self.HEADER:self.HEADER+v.height]=plane
        cv2.putText(canvas,f"BEV DETAIL | {v.pixels_per_meter:g} px / m on BOTH axes | 1 m grid | frame {record['frame_index']}",
                    (16,27),0,.7,(225,240,230),2,cv2.LINE_AA)
        notice=('OFFLINE RTS | Uses future frames; same IDs / fixed-size policy | Calibration remains unvalidated'
                if any(o.get('offline_estimate') for o in record['objects']) else
                'DISPLAY ZOOM ONLY | Coordinates and physical dimensions unchanged | Calibration remains unvalidated')
        cv2.putText(canvas,notice,
                    (16,51),0,.52,(95,205,245),1,cv2.LINE_AA)
        for y in range(math.ceil(v.y_min/10)*10,math.floor(v.y_max/10)*10+1,10):
            x,_=v.point([v.x_min,y])
            cv2.putText(canvas,f'{y}m',(max(0,min(x-17,v.width-65)),self.HEADER+20),0,.5,(240,240,240),1,cv2.LINE_AA)
        start=(24,v.height+self.HEADER+24); s=v.pixels_per_meter
        cv2.line(canvas,start,(round(start[0]+10*s),start[1]),(245,245,245),2)
        for n in range(11):
            x=round(start[0]+n*s)
            cv2.line(canvas,(x,start[1]-5),(x,start[1]+5),(245,245,245),1)
            if n in (0,1,5,10):
                cv2.putText(canvas,f'{n}m',(x-6,start[1]+24),0,.38,(240,240,240),1)
        cv2.putText(canvas,f"Located: {record['localized_count']}  |  P = predicted; S = offline smoothed  |  Nominal metres, not surveyed truth",
                    (round(10*s+70),start[1]+7),0,.53,(210,220,215),1,cv2.LINE_AA)
        return canvas
