"""Experimental RGB/geometry global association. No CARLA dependency or actor IDs."""
import math
import numpy as np

class GlobalTracker:
    def __init__(self, max_gap_s=2.):
        self.max_gap=max_gap_s;self.tracks={};self.local={};self.next_id=1

    def update(self, observations, time_s):
        seen={};accepted=[]
        for obs in sorted(observations,key=lambda o:(o['camera'],o['local_id'])):
            key=(obs['camera'],obs['local_id']);gid=self.local.get(key)
            if gid in self.tracks and time_s-self.tracks[gid]['last_time']>self.max_gap:gid=None
            if gid is None:
                candidates=[]
                for tid,t in self.tracks.items():
                    if obs['camera'] in seen.get(tid,set()):continue
                    # Cross-camera handoff only. Local tracking owns same-camera identity.
                    if t['camera']==obs['camera']:continue
                    dt=time_s-t['last_time']
                    if not 0<=dt<=self.max_gap:continue
                    predicted=t['xy']+t['velocity']*dt
                    distance=float(np.linalg.norm(obs['xy']-predicted))
                    cosine=float(np.dot(obs['appearance'],t['appearance']))
                    angle=abs(math.atan2(math.sin(obs['heading']-t['heading']),math.cos(obs['heading']-t['heading'])))
                    if distance<2.5 and cosine>.82 and angle<math.pi/4 and obs['class_name']==t['class_name']:
                        candidates.append((distance+(1-cosine)*3,tid))
                candidates.sort()
                if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>.5):gid=candidates[0][1]
                else:gid=self.next_id;self.next_id+=1
                self.local[key]=gid
            seen.setdefault(gid,set()).add(obs['camera'])
            accepted.append({**obs,'global_id':gid})
            self.tracks[gid]={**obs,'last_time':time_s}
        groups={}
        for obs in accepted:groups.setdefault(obs['global_id'],[]).append(obs)
        output=[]
        for gid,items in groups.items():
            # Weight estimates by predicted ground covariance, never by true position.
            weights=np.array([1/max(o['variance'],.05) for o in items]);weights/=weights.sum()
            first=items[0];xy=sum(o['xy']*w for o,w in zip(items,weights))
            self.tracks[gid]['xy']=xy
            output.append({'id':gid,'xy':xy.tolist(),'heading':first['heading'],'dimensions':first['dimensions'],
                'class_name':first['class_name'],'cameras':sorted({o['camera'] for o in items}),
                'sources':[{'camera':o['camera'],'local_id':o['local_id']} for o in items],
                'status':'observed' if any(o['observed'] for o in items) else 'predicted',
                'association':'experimental_image_geometry_gated'})
        for gid in list(self.tracks):
            if time_s-self.tracks[gid]['last_time']>self.max_gap:del self.tracks[gid]
        # Keep local->global mapping only for live tracks.
        self.local={k:v for k,v in self.local.items() if v in self.tracks}
        return output
