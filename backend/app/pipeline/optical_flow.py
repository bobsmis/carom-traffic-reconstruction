from __future__ import annotations

import cv2
import numpy as np


def sparse_vehicle_flow(
    previous_bgr: np.ndarray | None,
    current_bgr: np.ndarray,
    box: np.ndarray,
    max_corners: int = 28,
) -> list[list[float]]:
    """Track sparse image features inside one expanded vehicle ROI."""
    if previous_bgr is None:
        return []
    previous = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    roi = np.zeros(previous.shape, dtype=np.uint8)
    left = max(0, round(x1 - width * 0.35))
    top = max(0, round(y1 - height * 0.35))
    right = min(previous.shape[1] - 1, round(x2 + width * 0.35))
    bottom = min(previous.shape[0] - 1, round(y2 + height * 0.35))
    if right <= left or bottom <= top:
        return []
    cv2.rectangle(roi, (left, top), (right, bottom), 255, -1)
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=max_corners,
        qualityLevel=0.025,
        minDistance=4,
        mask=roi,
        blockSize=5,
    )
    if points is None:
        return []
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(17, 17),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    if moved is None or status is None:
        return []
    vectors = []
    for start, end, valid in zip(points.reshape(-1, 2), moved.reshape(-1, 2), status.reshape(-1), strict=True):
        distance = float(np.linalg.norm(end - start))
        if not valid or distance < 0.25 or distance > 45.0:
            continue
        vectors.append(
            [
                round(float(start[0]), 2),
                round(float(start[1]), 2),
                round(float(end[0]), 2),
                round(float(end[1]), 2),
            ]
        )
    return vectors
