"""Flow/mask instance association and observable/predicted metric states."""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .carom_geometry import masked_flow
from .vehicle_size import VehicleSize
from .geometry_recovery import GeometryRecovery


def polygon_iou(a, b):
    if len(a) < 3 or len(b) < 3:
        return 0.
    a = cv2.convexHull(np.asarray(a,np.float32))
    b = cv2.convexHull(np.asarray(b,np.float32))
    area, _ = cv2.intersectConvexConvex(a,b)
    return float(max(0,area)/max(cv2.contourArea(a)+cv2.contourArea(b)-area,1e-6))


def deduplicate_observations(detections, threshold=.72):
    """Class-compatible mask NMS BEFORE assigning identities, with audit reasons.

    A far vehicle can appear twice in class-aware detector NMS (bus + truck).
    This is duplicate image evidence, not BEV collision-based suppression.
    """
    kept=[]; rejected=[]
    compatible={'car','bus','truck'}
    for index in sorted(range(len(detections)),key=lambda i:-detections[i].get('confidence',0)):
        item=detections[index]; duplicate=None
        for other in kept:
            target=detections[other]
            if (item['class_name']!=target['class_name'] and
                    not {item['class_name'],target['class_name']}<=compatible):
                continue
            overlap=polygon_iou(item['mask_polygon'],target['mask_polygon'])
            if overlap>=threshold:
                duplicate={'candidate_index':index,'bbox_xyxy':item['bbox_xyxy'],
                           'eligible':False,'reason':'duplicate_instance_mask',
                           'kept_candidate_index':other,'mask_hull_iou':overlap}
                break
        if duplicate:
            rejected.append(duplicate)
        else:
            kept.append(index)
    kept.sort()
    return [detections[i] for i in kept],rejected,kept


@dataclass
class MetricState:
    mean: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.diag([1.,1.,100.,100.]))

    @classmethod
    def at(cls, xy):
        return cls(np.r_[xy,0.,0.].astype(float))

    def predict(self, dt):
        F = np.eye(4); F[:2,2:] = np.eye(2)*dt
        G = np.vstack((np.eye(2)*dt**2/2, np.eye(2)*dt))
        self.mean = F@self.mean
        self.covariance = F@self.covariance@F.T + G@G.T*9

    def observe(self, value, *, velocity=False, variance=1.):
        H = np.zeros((2,4)); H[:,2 if velocity else 0:4 if velocity else 2] = np.eye(2)
        R = np.eye(2)*variance
        innovation = np.asarray(value)-H@self.mean
        S = H@self.covariance@H.T+R
        if innovation@np.linalg.solve(S,innovation) > 16.:
            return False
        K = np.linalg.solve(S,H@self.covariance).T
        self.mean += K@innovation
        I = np.eye(4)-K@H
        self.covariance = I@self.covariance@I.T+K@R@K.T
        return True


@dataclass
class Instance:
    id: int
    detection: dict
    polygon: np.ndarray
    lost: int = 0
    age: int = 1
    image_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    flow: np.ndarray = field(default_factory=lambda: np.empty((0,4)))
    heading: float | None = None
    heading_source: str = 'pending'
    metric: MetricState | None = None
    geometry_age: int = 0
    metric_observations: int = 0
    geometry_epoch: int = 0
    geometry_reject_streak: int = 0
    size_class: str | None = None
    recovery: GeometryRecovery = field(default_factory=GeometryRecovery)
    fit: dict | None = None
    initial_center: np.ndarray | None = None
    class_evidence: dict = field(default_factory=dict)
    size: VehicleSize = field(default_factory=VehicleSize)
    hits: int = 1
    hit_streak: int = 1
    confirmed: bool = False
    appearance: np.ndarray | None = None
    association_score: float | None = None
    reacquired_after_frames: int = 0

    def __post_init__(self):
        self.class_evidence[self.detection['class_name']]=max(.1,self.detection.get('confidence',1.))

    @property
    def class_name(self):
        return max(self.class_evidence,key=self.class_evidence.get)

    def observe_class(self,detection):
        for key in self.class_evidence:
            self.class_evidence[key]*=.97
        name=detection['class_name']
        self.class_evidence[name]=self.class_evidence.get(name,0)+max(.1,detection.get('confidence',1.))


def appearance_histogram(image, polygon):
    """Local masked HSV evidence, not an identity oracle or lane constraint."""
    if image is None or len(polygon)<3:
        return None
    h,w=image.shape[:2]
    p=np.asarray(polygon,float)
    lo=np.maximum(np.floor(p.min(axis=0)).astype(int),0)
    hi=np.minimum(np.ceil(p.max(axis=0)).astype(int)+1,[w,h])
    x,y=lo; xx,yy=hi
    if xx<=x or yy<=y:
        return None
    mask=np.zeros((yy-y,xx-x),np.uint8)
    cv2.fillPoly(mask,[np.round(p-lo).astype(np.int32)],255)
    hsv=cv2.cvtColor(image[y:yy,x:xx],cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],mask,[8,4,4],[0,180,0,256,0,256]).ravel()
    return hist/max(float(hist.sum()),1.)


class FlowMaskTracker:
    """Persistent, non-recycled IDs. No imported ByteTrack IDs or world points."""
    def __init__(self, max_lost=18, min_hits=1, tentative_max_lost=1):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        if max_lost<0 or min_hits<1 or tentative_max_lost<0:
            raise ValueError('invalid lifecycle parameters')
        self.min_hits=min_hits
        self.tentative_max_lost=tentative_max_lost

    def update(self, detections, previous_gray, gray, dt, current_bgr=None):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('positive finite dt required')
        appearances=[appearance_histogram(current_bgr,d['mask_polygon']) for d in detections]
        old = list(self.tracks.values())
        predicted = []
        for track in old:
            track.reacquired_after_frames=0
            track.age += 1
            track.geometry_age += 1
            if track.metric is not None:
                track.metric.predict(dt)
            track.flow = (masked_flow(previous_gray,gray,track.polygon)
                          if track.lost == 0 else np.empty((0,4)))
            shift = np.median(track.flow[:,2:]-track.flow[:,:2],axis=0) if len(track.flow) >= 3 else track.image_velocity
            predicted.append(track.polygon+shift)
        scores = np.zeros((len(old),len(detections)))
        for i,track in enumerate(old):
            for j,d in enumerate(detections):
                compatible={'car','bus','truck'}
                same_class=d['class_name']==track.class_name
                if not same_class and not {d['class_name'],track.class_name}<=compatible:
                    continue
                polygon = np.asarray(d['mask_polygon'],float)
                overlap = polygon_iou(predicted[i],polygon)
                votes = 0.
                if len(track.flow) >= 3 and len(polygon) >= 3:
                    contour = polygon.astype(np.float32)
                    votes = np.mean([cv2.pointPolygonTest(contour,tuple(p),False) >= 0 for p in track.flow[:,2:]])
                score=.65*overlap+.35*votes if len(track.flow) >= 3 else overlap
                hist=appearances[j]
                if track.appearance is not None and hist is not None:
                    similarity=float(np.sqrt(track.appearance*hist).sum())
                    # Only relax mask-overlap matching for temporarily lost,
                    # confirmed tracks, inside a size-normalized motion gate.
                    extent=np.maximum(np.ptp(predicted[i],axis=0),5.)
                    distance=np.linalg.norm((polygon.mean(axis=0)-predicted[i].mean(axis=0))/extent)
                    area_ratio=abs(cv2.contourArea(polygon.astype(np.float32)))/max(abs(cv2.contourArea(predicted[i].astype(np.float32))),1.)
                    if track.lost and track.confirmed and similarity>=.88 and distance<=.75 and .5<=area_ratio<=2:
                        score=max(score,.4*similarity*np.exp(-distance**2))
                    if similarity<.35:
                        score*=.25
                # YOLO class jitter is not evidence of a different identity.
                scores[i,j]=score*(1. if same_class else .9)
        matched_old, matched_new = set(),set()
        if scores.size:
            rows,cols=linear_sum_assignment(-scores)
            for i,j in zip(rows,cols):
                if scores[i,j] < .18:
                    continue
                track=old[i]; d=detections[j]
                polygon=np.asarray(d['mask_polygon'],float)
                # Current track polygon already propagated on every missed frame.
                delta=polygon.mean(axis=0)-track.polygon.mean(axis=0)
                if len(track.flow) >= 3:
                    delta=np.median(track.flow[:,2:]-track.flow[:,:2],axis=0)
                    contour=polygon.astype(np.float32)
                    track.flow=np.asarray([f for f in track.flow if cv2.pointPolygonTest(contour,tuple(f[2:]),False)>=0]).reshape(-1,4)
                track.image_velocity=.5*track.image_velocity+.5*delta
                track.reacquired_after_frames=track.lost
                track.hits+=1; track.hit_streak+=1
                track.confirmed=track.confirmed or track.hit_streak>=self.min_hits
                track.association_score=float(scores[i,j])
                if appearances[j] is not None:
                    track.appearance=(appearances[j] if track.appearance is None else
                                      .9*track.appearance+.1*appearances[j])
                track.polygon=polygon; track.observe_class(d); track.detection=d; track.lost=0
                matched_old.add(i); matched_new.add(j)
        retired=[]
        for i,track in enumerate(old):
            if i not in matched_old:
                track.lost+=1; track.hit_streak=0; track.polygon=predicted[i]
                limit=self.max_lost if track.confirmed else self.tentative_max_lost
                if track.lost > limit:
                    retired.append(track.id)
        for tid in retired:
            del self.tracks[tid]
        for j,d in enumerate(detections):
            if j in matched_new:
                continue
            polygon=np.asarray(d['mask_polygon'],float)
            track=Instance(self.next_id,d,polygon,initial_center=polygon.mean(axis=0))
            track.confirmed=self.min_hits==1
            track.appearance=appearances[j]
            self.tracks[track.id]=track; self.next_id+=1
        return list(self.tracks.values()),retired
