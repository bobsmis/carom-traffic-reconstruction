"""Scene-specific measurement eligibility, not a calibration or lane correction.

The silhouette foot is used ONLY for conservative ROI preselection. Published
positions still come from the fitted cuboid centre and are gated a second time.
All metric limits inherit calibration uncertainty.
"""
import cv2
import numpy as np


class MeasurementRegion:
    def __init__(self, camera, config):
        self.camera = camera
        self.settings = dict(config.get('measurement_region', {}))
        self.enabled = self.settings.get('enabled', False)
        self.polygon = np.asarray(self.settings.get('ground_polygon_m', []), np.float32)
        if self.enabled:
            if (self.polygon.ndim != 2 or self.polygon.shape[1] != 2 or len(self.polygon) < 3
                    or not np.isfinite(self.polygon).all() or not cv2.isContourConvex(self.polygon)):
                raise ValueError('measurement region must be a finite convex ground polygon')
            for key in ('max_camera_ground_distance_m', 'min_bbox_width_px', 'min_bbox_height_px',
                        'min_mask_area_px', 'pixel_noise_sigma_px', 'max_projection_sigma_m'):
                if not np.isfinite(self.settings[key]) or self.settings[key] <= 0:
                    raise ValueError(f'invalid measurement threshold: {key}')

    def image_polygon(self):
        return self.camera.project(np.c_[self.polygon, np.zeros(len(self.polygon))])

    def ground_check(self, xy):
        if not self.enabled:
            return {'eligible': True}
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return {'eligible': False, 'reason': 'nonfinite_ground'}
        origin = -self.camera.R.T @ self.camera.t
        distance = float(np.linalg.norm(xy-origin[:2]))
        result = {'eligible': False, 'camera_ground_distance_m': distance}
        if distance > self.settings['max_camera_ground_distance_m']:
            return dict(result, reason='beyond_camera_range')
        if cv2.pointPolygonTest(self.polygon, tuple(xy), False) < 0:
            return dict(result, reason='outside_measurement_region')
        try:
            uv = self.camera.project([[*xy, 0.]])[0]
            # Numerical derivative of plane intersection; conditional pixel-noise
            # sensitivity only. Calibration uncertainty is NOT included.
            q = self.camera.ground([uv+[.5,0], uv-[.5,0], uv+[0,.5], uv-[0,.5]])
            J = np.column_stack((q[0]-q[1], q[2]-q[3]))
            sigma = float(np.linalg.svd(J, compute_uv=False)[0]*self.settings['pixel_noise_sigma_px'])
        except (ValueError, np.linalg.LinAlgError):
            return dict(result, reason='invalid_ground_projection')
        result['conditional_projection_sigma_m'] = sigma
        if not np.isfinite(sigma) or sigma > self.settings['max_projection_sigma_m']:
            return dict(result, reason='projection_too_sensitive')
        return dict(result, eligible=True)

    def observation_check(self, detection):
        if not self.enabled:
            return {'eligible': True}
        if detection.get('detector_crop_truncated', False):
            return {'eligible': False, 'reason': 'detector_crop_truncation'}
        box = np.asarray(detection['bbox_xyxy'], float)
        mask = np.asarray(detection['mask_polygon'], float)
        if (box.shape != (4,) or mask.ndim != 2 or mask.shape[1] != 2 or len(mask) < 3
                or not np.isfinite(box).all() or not np.isfinite(mask).all()):
            return {'eligible': False, 'reason': 'invalid_observation'}
        width, height = box[2:]-box[:2]
        area = float(abs(cv2.contourArea(mask.astype(np.float32))))
        if width < self.settings['min_bbox_width_px'] or height < self.settings['min_bbox_height_px']:
            return {'eligible': False, 'reason': 'insufficient_pixel_extent'}
        if area < self.settings['min_mask_area_px']:
            return {'eligible': False, 'reason': 'insufficient_mask_area'}
        bottom = mask[:,1].max()
        band = mask[mask[:,1] >= bottom-max(2., .08*height)]
        foot = [float(np.median(band[:,0])), float(bottom)]
        try:
            result = self.ground_check(self.camera.ground([foot])[0])
        except (ValueError, np.linalg.LinAlgError):
            result = {'eligible': False, 'reason': 'invalid_ground_projection'}
        return dict(result, roi_probe_px=foot, mask_area_px=area,
                    probe_role='preselection_only_not_vehicle_position')

    def detector_crop(self, shape):
        """Bounding crop with context for the detector; polygon gating follows.

        Do not black out the polygon: cutting a vehicle silhouette at the ROI
        edge would manufacture geometry. Return offsets for exact restoration.
        """
        h,w = shape[:2]
        if not self.enabled:
            return 0,0,w,h
        p = self.image_polygon()
        pad = int(self.settings.get('detector_context_px', 96))
        x1,y1 = np.maximum(0, np.floor(p.min(axis=0)).astype(int)-pad)
        x2,y2 = np.minimum([w,h], np.ceil(p.max(axis=0)).astype(int)+pad)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('measurement region does not intersect video')
        return int(x1),int(y1),int(x2),int(y2)

    def split(self, detections):
        kept, rejected = [], []
        for i,d in enumerate(detections):
            result = self.observation_check(d)
            if result['eligible']:
                kept.append(d)
            else:
                rejected.append({'candidate_index':i,'bbox_xyxy':d['bbox_xyxy'], **result})
        return kept,rejected
