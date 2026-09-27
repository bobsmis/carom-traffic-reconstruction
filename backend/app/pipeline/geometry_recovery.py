"""Independent multi-frame evidence for provisional geometry recovery.

No ID merging, lane snapping or collision-based relocation. Candidate evidence
survives rejection by the old metric filter, breaking the bootstrap deadlock.
"""
from dataclasses import dataclass,field
import numpy as np


@dataclass
class GeometryRecovery:
    samples: list = field(default_factory=list)

    def observe(self, fit, class_name, frame, association_score, eligible=True):
        good=(eligible and fit and fit.get('status')=='fitted'
              and fit.get('silhouette_hull_iou',0)>=.8 and fit.get('support_rmse_m',99)<=.2
              and fit.get('support_rank',0)>=5 and (association_score or 0)>=.35)
        if not good:
            self.samples.clear(); return
        dims=np.asarray(fit['dimensions_lwh'],float); xy=np.asarray(fit['ground_xy'],float)
        if dims.shape!=(3,) or xy.shape!=(2,) or not np.isfinite(np.r_[dims,xy]).all() or np.any(dims<=0):
            self.samples.clear();return
        if self.samples and (frame!=self.samples[-1]['frame']+1 or class_name!=self.samples[-1]['class_name']):
            self.samples.clear()
        self.samples.append({'frame':frame,'class_name':class_name,'dimensions':dims.tolist(),'xy':xy.tolist()})
        self.samples=self.samples[-12:]

    def candidate(self, fps, minimum=6):
        if len(self.samples)<minimum: return None
        samples=self.samples[-minimum:]
        dims=np.array([s['dimensions'] for s in samples]);median=np.median(dims,axis=0)
        if np.max(np.abs(dims/median-1))>.15: return None
        t=(np.array([s['frame'] for s in samples])-samples[-1]['frame'])/fps
        xy=np.array([s['xy'] for s in samples])
        design=np.c_[np.ones(len(t)),t]
        parameters=np.linalg.lstsq(design,xy,rcond=None)[0]
        residual=float(np.sqrt(np.mean(np.sum((design@parameters-xy)**2,axis=1))))
        if residual>.6 or np.linalg.norm(parameters[1])>45.: return None
        return {'dimensions_lwh':median.tolist(),'xy_m':parameters[0].tolist(),
                'velocity_xy_mps':parameters[1].tolist(),'motion_residual_m':residual,
                'sample_count':len(samples),'class_name':samples[-1]['class_name'],
                'first_frame':samples[0]['frame'],'last_frame':samples[-1]['frame']}
