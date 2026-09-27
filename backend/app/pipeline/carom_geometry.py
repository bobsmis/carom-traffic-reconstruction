"""Paper-guided CAROM geometry, independent of the legacy display pipeline.

All image geometry uses ONE pinhole projection. No lane snapping, silhouette
bottom-point localisation, or screen-space fabricated cuboids live here.
This is a reimplementation, not the unavailable authors' implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class RoadCamera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self):
        return self.K @ np.column_stack((self.R, self.t))

    @property
    def H(self):
        return self.P[:, [0, 1, 3]]

    @property
    def horizon(self):
        h = np.linalg.inv(self.H).T @ np.array([0., 0., 1.])
        return h / np.linalg.norm(h[:2])

    def project(self, xyz):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        q = np.column_stack((xyz, np.ones(len(xyz)))) @ self.P.T
        if np.any(q[:, 2] <= 1e-6):
            raise ValueError("point behind camera")
        return q[:, :2] / q[:, 2:3]

    def ground(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        q = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(self.H).T
        if np.any(np.abs(q[:, 2]) < 1e-8):
            raise ValueError("point on horizon")
        result = q[:, :2] / q[:, 2:3]
        self.project(np.column_stack((result, np.zeros(len(result)))))
        return result

    @classmethod
    def from_scene(cls, config, *, experimental=False):
        """Fit rigid pose with fixed K; never silently manufacture calibration.

        Strict use needs >=6 surveyed correspondences and independent check
        points. A held-out residual is a consistency check, not GPS accuracy.
        Inputs must already be undistorted (the paper's pinhole assumption).
        """
        c = config['planar_calibration']
        if c.get('method') == 'carla_exact_extrinsics':
            K=np.asarray(c['intrinsic_matrix'],float)
            E=np.asarray(c['world_to_camera_cv'],float)
            if (K.shape!=(3,3) or E.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(E).all()
                    or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1])
                    or not np.allclose(E[3],[0,0,0,1])
                    or not np.allclose(E[:3,:3].T@E[:3,:3],np.eye(3),atol=1e-6)
                    or not np.isclose(np.linalg.det(E[:3,:3]),1.,atol=1e-6)):
                raise ValueError('invalid simulator camera matrices')
            if np.any(np.abs(np.asarray(c.get('distortion',[0]),float))>1e-10):
                raise ValueError('simulator input must have zero lens distortion')
            camera=cls(K,E[:3,:3],E[:3,3])
            if abs(np.linalg.det(camera.H))<1e-9:raise ValueError('degenerate simulator ground projection')
            return camera,{'status':'simulation_parameters_provided','surveyed':False,
                'intrinsic_source':'carla_sensor_configuration','control_count':0,
                'control_reprojection_rmse_px':None,'control_reprojection_max_px':None,'check_rmse_px':None,
                'same_projection_for_ground_and_cuboid':True,'projection_matrix':camera.P.tolist(),
                'note':'Exact simulator configuration; not a real-world calibration or an independent accuracy test.'}
        im = np.asarray(c['image_points_px'], np.float64)
        xy = np.asarray(c['ground_points_m'], np.float64)
        width, height = config['crop']['width'], config['crop']['height']
        explicit_k = 'intrinsic_matrix' in c
        f = c.get('focal_length_px', max(width, height))
        K = np.asarray(c.get('intrinsic_matrix', [[f, 0, width/2], [0, f, height/2], [0, 0, 1]]), float)
        if im.ndim!=2 or im.shape[1:]!=(2,) or len(im) < 4 or im.shape != xy.shape or not np.isfinite(im).all() or not np.isfinite(xy).all():
            raise ValueError('invalid calibration pairs')
        if (K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0
                or not np.allclose(K[2],[0,0,1]) or abs(np.linalg.det(K))<1e-9):
            raise ValueError('invalid intrinsic matrix')
        if np.linalg.matrix_rank(xy-xy.mean(axis=0))<2 or np.linalg.matrix_rank(im-im.mean(axis=0))<2:
            raise ValueError('collinear calibration pairs')
        if len(np.unique(xy,axis=0))!=len(xy) or len(np.unique(im,axis=0))!=len(im):
            raise ValueError('duplicate calibration controls')
        if np.any(np.abs(np.asarray(c.get('distortion', [0]), float)) > 1e-10):
            raise ValueError('undistort video and calibration points before this pinhole pipeline')
        checks = c.get('check_points', {})
        if checks:
            ci=np.asarray(checks.get('image_points_px',[]),float)
            cw=np.asarray(checks.get('ground_points_m',[]),float)
            if ci.ndim!=2 or ci.shape[1:]!=(2,) or ci.shape!=cw.shape or len(ci)<2 or not np.isfinite(ci).all() or not np.isfinite(cw).all():
                raise ValueError('invalid independent check points')
            if len(np.unique(ci,axis=0))!=len(ci) or len(np.unique(cw,axis=0))!=len(cw):
                raise ValueError('duplicate independent check points')
            if np.any(np.linalg.norm(cw[:,None]-xy[None],axis=2)<1e-5) or np.any(np.linalg.norm(ci[:,None]-im[None],axis=2)<.01):
                raise ValueError('independent checks must not reuse calibration controls')
        ready = (len(im) >= 6 and explicit_k and c.get('surveyed', False)
                 and len(checks.get('image_points_px', [])) >= 2)
        if not experimental and not ready:
            raise ValueError('strict calibration requires measured K, >=6 surveyed pairs and >=2 independent check points')
        obj = np.column_stack((xy, np.zeros(len(xy))))
        ok, rvec, tvec = cv2.solvePnP(obj, im, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError('pose estimation failed')
        rvec, tvec = cv2.solvePnPRefineLM(obj, im, K, None, rvec, tvec)
        camera = cls(K, cv2.Rodrigues(rvec)[0], tvec.ravel())
        residual = np.linalg.norm(camera.project(obj) - im, axis=1)
        audit = {'status': 'experimental_unverified', 'surveyed': bool(c.get('surveyed', False)),
                 'intrinsic_source': 'provided' if explicit_k else 'assumed_focal_length',
                 'control_count': len(im), 'control_reprojection_rmse_px': float(np.sqrt(np.mean(residual**2))),
                 'control_reprojection_max_px': float(residual.max()), 'check_rmse_px': None,
                 'same_projection_for_ground_and_cuboid': True, 'projection_matrix': camera.P.tolist()}
        if checks:
            p = np.asarray(checks['ground_points_m'], float)
            error = np.linalg.norm(camera.project(np.column_stack((p, np.zeros(len(p))))) - checks['image_points_px'], axis=1)
            audit['check_rmse_px'] = float(np.sqrt(np.mean(error**2)))
            if (ready and audit['check_rmse_px'] <= c.get('max_check_rmse_px', 3.)
                    and audit['control_reprojection_rmse_px']<=c.get('max_control_rmse_px',3.)):
                audit['status'] = 'calibration_check_passed'
        if not experimental and audit['status'] != 'calibration_check_passed':
            raise ValueError('independent calibration check exceeds tolerance')
        return camera, audit


def masked_flow(previous, current, polygon, max_points=70):
    """Previous-instance features, pyramidal LK, forward/backward validation."""
    if previous is None or len(polygon) < 3:
        return np.empty((0, 4))
    mask = np.zeros(previous.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(previous, max_points, .015, 3, mask=mask, blockSize=5)
    if points is None:
        return np.empty((0, 4))
    kw = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    moved, s1, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **kw)
    if moved is None:
        return np.empty((0, 4))
    back, s2, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, **kw)
    if back is None:
        return np.empty((0, 4))
    a, b = points.reshape(-1, 2), moved.reshape(-1, 2)
    valid = ((s1.ravel() == 1) & (s2.ravel() == 1)
             & (np.linalg.norm(back.reshape(-1, 2) - a, axis=1) < 1.)
             & (np.linalg.norm(b-a, axis=1) < 65.) & np.isfinite(b).all(axis=1))
    return np.column_stack((a[valid], b[valid])).astype(float)


def flow_heading(camera, vectors, min_inliers=5, threshold_deg=4.):
    """Constrained vanishing-point RANSAC; homogeneous VPs include infinity.

    Enumerate one-line minimal hypotheses on the calibrated horizon instead
    of random sampling. Select consensus by angular line residual, then refine
    the ground direction with SVD. Sign comes from observed optical flow.
    """
    f = np.asarray(vectors, float).reshape(-1, 4)
    f = f[np.linalg.norm(f[:, 2:]-f[:, :2], axis=1) > .35]
    if len(f) < min_inliers:
        return None
    a = np.column_stack((f[:, :2], np.ones(len(f))))
    b = np.column_stack((f[:, 2:], np.ones(len(f))))
    lines = np.cross(a, b)
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    vps = np.cross(lines, camera.horizon)
    flow = f[:, 2:] - f[:, :2]
    flow /= np.linalg.norm(flow, axis=1)[:, None]

    def residual(vp):
        d = vp[:2] - f[:, :2]*vp[2]
        norm = np.linalg.norm(d, axis=1)
        cross = np.abs(d[:, 0]*flow[:, 1] - d[:, 1]*flow[:, 0])
        return cross / np.maximum(norm, 1e-9)

    threshold = math.sin(math.radians(threshold_deg))
    masks = [residual(v) < threshold for v in vps]
    good = max(masks, key=np.count_nonzero)
    if good.sum() < min_inliers or good.mean() < .5:
        return None
    equations = lines[good] @ camera.P[:, :2]
    equations /= np.maximum(np.linalg.norm(equations, axis=1)[:, None], 1e-9)
    _, _, vt = np.linalg.svd(equations)
    direction = vt[-1]
    vp = camera.P[:, :2] @ direction
    good = residual(vp) < threshold
    if good.sum() < min_inliers or good.mean() < .5:
        return None
    # Projection derivative sign is vp.xy - pixel*vp.z (positive camera depth).
    d = vp[:2] - f[good, :2]*vp[2]
    if np.median(np.sum(d * flow[good], axis=1)) < 0:
        direction = -direction
    return {'heading_rad': float(math.atan2(direction[1], direction[0])),
            'inlier_count': int(good.sum()), 'candidate_count': len(f),
            'inlier_flow': f[good].tolist(), 'method': 'horizon_constrained_flow_consensus'}


def cuboid_vertices(center, heading, dimensions):
    L, W, H = dimensions
    a = np.array([math.cos(heading), math.sin(heading)])
    b = np.array([-a[1], a[0]])
    base = np.array([np.asarray(center) + s*L/2*a + t*W/2*b
                     for s, t in ((1,1),(1,-1),(-1,-1),(-1,1))])
    return np.vstack((np.column_stack((base, np.zeros(4))), np.column_stack((base, np.full(4,H)))))


def tangent_lines(polygon, vp):
    hull = cv2.convexHull(np.asarray(polygon, np.float32)).reshape(-1, 2).astype(float)
    hp = np.column_stack((hull, np.ones(len(hull))))
    candidates = np.cross(np.broadcast_to(vp, hp.shape), hp)
    result = []
    for line in candidates:
        n = np.linalg.norm(line[:2])
        if n < 1e-8:
            continue
        line /= n
        distances = hp @ line
        if distances.max() <= 1e-5:
            pass
        elif distances.min() >= -1e-5:
            line = -line
        else:
            continue
        if not any(np.linalg.norm(line - old) < 1e-3 for old in result):
            result.append(line.copy())
    return result


def fit_cuboid(camera, polygon, heading, prior, min_iou=.45, fixed_dimensions=None):
    """Fit three-VP silhouette support planes with bounded class-size priors.

    Unlike a generic 2D bbox extrusion, all six support planes are backprojected
    into world space. Solve [cx,cy,L,W,H] with base constrained to z=0.
    Regularisation is an explicit deviation from the paper's empirical sizing.
    Degenerate silhouettes remain prior-dependent and are labelled accordingly.
    """
    polygon = np.asarray(polygon, float).reshape(-1, 2)
    if len(polygon) < 4 or abs(cv2.contourArea(polygon.astype(np.float32))) < 50:
        return {'status': 'rejected', 'reason': 'insufficient_silhouette'}
    a = np.array([math.cos(heading), math.sin(heading), 0.])
    b = np.array([-a[1], a[0], 0.])
    lines = []
    for axis in (a, b, np.array([0., 0., 1.])):
        found = tangent_lines(polygon, camera.P[:, :3] @ axis)
        if len(found) != 2:
            return {'status': 'rejected', 'reason': 'degenerate_vanishing_tangents'}
        lines.extend(found)
    planes = np.asarray(lines) @ camera.P
    planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
    A = np.column_stack((planes[:, 0], planes[:, 1], np.abs(planes[:, :3]@a)/2,
                         np.abs(planes[:, :3]@b)/2, (planes[:, 2]+np.abs(planes[:, 2]))/2))
    rhs = -planes[:, 3]
    prior = np.asarray(prior, float)
    # Class bounds are explicit priors, not measurements of invisible surfaces.
    regularizer = np.zeros((3, 5))
    regularizer[:, 2:] = np.diag(.025/prior)
    if fixed_dimensions is None:
        solution = lsq_linear(np.vstack((A, regularizer)), np.r_[rhs, np.full(3, .025)],
                              bounds=(np.r_[[-500., -500.], prior*[.55,.65,.55]],
                                      np.r_[[500.,1000.], prior*[1.6,1.4,1.6]]), tol=1e-10)
        parameters = solution.x
    else:
        dims = np.asarray(fixed_dimensions, float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            raise ValueError('fixed dimensions must be three finite positive lengths')
        # Same silhouette support planes; only ground translation is fitted.
        solution = lsq_linear(A[:, :2], rhs-A[:, 2:]@dims,
                              bounds=([-500., -500.], [500., 1000.]), tol=1e-10)
        parameters = np.r_[solution.x, dims]
    cx, cy, L, W, H = parameters
    try:
        corners = camera.project(cuboid_vertices([cx,cy], heading, [L,W,H]))
    except ValueError:
        return {'status': 'rejected', 'reason': 'nonpositive_depth'}
    hull = cv2.convexHull(polygon.astype(np.float32))
    envelope = cv2.convexHull(corners.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(hull, envelope)
    iou = float(inter / max(cv2.contourArea(hull)+cv2.contourArea(envelope)-inter, 1e-6))
    residual = float(np.sqrt(np.mean((A @ parameters-rhs)**2)))
    if not solution.success or not np.isfinite(parameters).all() or iou < min_iou or residual > .45:
        return {'status': 'rejected', 'reason': 'silhouette_fit_quality', 'silhouette_hull_iou': iou,
                'support_rmse_m': residual}
    return {'status': 'fitted', 'ground_xy': [cx,cy], 'dimensions_lwh': [L,W,H],
            'heading_rad': heading, 'corners_image': corners.tolist(),
            'silhouette_hull_iou': iou, 'support_rmse_m': residual,
            'support_rank': int(np.linalg.matrix_rank(A)),
            'size_prior_used': True, 'dimensions_fixed': fixed_dimensions is not None,
            'method': 'three_vanishing_points_silhouette_support_planes'}


def flow_velocity(camera, fit, vectors, dt):
    """Ray/cuboid surface intersections give per-feature depths, then velocity.

    This replaces the paper's windowed velocity approximation with a pose-depth
    least-squares observation. It assumes rigid translation during one frame;
    it is not used as an independent ground-truth measurement.
    """
    f = np.asarray(vectors, float).reshape(-1,4)
    if len(f) < 5 or dt <= 0:
        return None
    theta = fit['heading_rad']
    rot = np.array([[math.cos(theta),-math.sin(theta),0], [math.sin(theta),math.cos(theta),0], [0,0,1.]])
    dims = np.asarray(fit['dimensions_lwh'])
    center = np.r_[fit['ground_xy'], dims[2]/2]
    origin = -camera.R.T @ camera.t
    local_origin = rot.T @ (origin-center)
    samples = []
    for u,v,s,t in f:
        ray = camera.R.T @ (np.linalg.inv(camera.K) @ np.array([u,v,1.]))
        dr = rot.T @ ray
        near, far = -np.inf, np.inf
        for k in range(3):
            if abs(dr[k]) < 1e-9:
                if abs(local_origin[k]) > dims[k]/2:
                    far = -np.inf
                    break
                continue
            limits = (-dims[k]/2-local_origin[k])/dr[k], (dims[k]/2-local_origin[k])/dr[k]
            near, far = max(near,min(limits)), min(far,max(limits))
        if far < max(near,0) or near <= 0:
            continue
        xyz = origin + near*ray
        q = camera.P @ np.r_[xyz,1.]
        J = (camera.P[:2,:2]*q[2] - q[:2,None]*camera.P[2,:2]) / q[2]**2
        if np.linalg.cond(J) > 100:
            continue
        samples.append(np.linalg.solve(J, np.array([s-u,t-v]))/dt)
    if len(samples) < 5:
        return None
    samples = np.asarray(samples)
    median = np.median(samples, axis=0)
    errors = np.linalg.norm(samples-median, axis=1)
    valid = errors < max(1.5, 3*np.median(errors))
    if valid.sum() < 5 or np.linalg.norm(median) > 60:
        return None
    return {'xy_mps': np.median(samples[valid], axis=0).tolist(), 'features': int(valid.sum()),
            'dispersion_mps': float(np.median(errors[valid])), 'method': 'flow_with_cuboid_ray_depth'}
