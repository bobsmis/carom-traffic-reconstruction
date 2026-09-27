from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetricCamera:
    """Pinhole camera tied to a planar metric road coordinate frame."""

    intrinsic: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    native_width: int
    native_height: int
    frame_width: int
    frame_height: int
    ground_to_image_override: np.ndarray | None = None

    @classmethod
    def from_tumtraf(
        cls,
        path: Path,
        frame_width: int,
        frame_height: int,
        transform_key: str,
    ) -> "MetricCamera":
        payload = json.loads(path.read_text(encoding="utf-8"))
        intrinsic = np.asarray(payload["intrinsic_camera_matrix"], dtype=float)[:, :3]
        transform = np.asarray(payload[transform_key], dtype=float)
        return cls(
            intrinsic=intrinsic,
            distortion=np.asarray(payload.get("dist_coefficients", []), dtype=float),
            world_to_camera=transform,
            native_width=int(payload["image_width"]),
            native_height=int(payload["image_height"]),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @classmethod
    def from_planar_correspondences(
        cls,
        image_points: list[list[float]],
        ground_points_m: list[list[float]],
        frame_width: int,
        frame_height: int,
        focal_length_px: float,
    ) -> "MetricCamera":
        """Build a metric road camera from configured pixel/ground pairs.

        The fitted ground-plane homography drives localisation. A pinhole pose
        decomposed from it is retained only for the optional cuboid projection.
        """
        if len(image_points) != len(ground_points_m) or len(image_points) < 4:
            raise ValueError("planar calibration needs at least four paired points")
        rows: list[list[float]] = []
        values: list[float] = []
        for (u, v), (x, y) in zip(image_points, ground_points_m, strict=True):
            rows.extend(
                (
                    [u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v],
                    [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v],
                )
            )
            values.extend((x, y))
        coefficients, *_ = np.linalg.lstsq(
            np.asarray(rows, dtype=float), np.asarray(values, dtype=float), rcond=None
        )
        image_to_ground = np.append(coefficients, 1.0).reshape(3, 3)
        ground_to_image = np.linalg.inv(image_to_ground)
        ground_to_image /= ground_to_image[2, 2]

        intrinsic = np.asarray(
            [
                [focal_length_px, 0.0, frame_width / 2.0],
                [0.0, focal_length_px, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        normalized = np.linalg.inv(intrinsic) @ ground_to_image
        scale = 2.0 / (
            np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1])
        )
        r1 = scale * normalized[:, 0]
        r2 = scale * normalized[:, 1]
        translation = scale * normalized[:, 2]
        rotation_guess = np.column_stack((r1, r2, np.cross(r1, r2)))
        left, _, right = np.linalg.svd(rotation_guess)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
            translation = -translation
        world_to_camera = np.eye(4, dtype=float)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation
        if (world_to_camera @ np.asarray([0.0, 20.0, 0.0, 1.0]))[2] < 0:
            world_to_camera[:3, :3] *= -1
            world_to_camera[:3, 3] *= -1

        return cls(
            intrinsic=intrinsic,
            distortion=np.zeros(5, dtype=float),
            world_to_camera=world_to_camera,
            native_width=frame_width,
            native_height=frame_height,
            frame_width=frame_width,
            frame_height=frame_height,
            ground_to_image_override=ground_to_image,
        )

    @classmethod
    def from_config(cls, root: Path, config: dict) -> "MetricCamera":
        crop = config["crop"]
        planar = config.get("planar_calibration")
        if planar is not None:
            return cls.from_planar_correspondences(
                image_points=planar["image_points_px"],
                ground_points_m=planar["ground_points_m"],
                frame_width=int(crop["width"]),
                frame_height=int(crop["height"]),
                focal_length_px=float(planar["focal_length_px"]),
            )
        return cls.from_tumtraf(
            root / config["calibration_path"],
            frame_width=int(crop["width"]),
            frame_height=int(crop["height"]),
            transform_key=config["ground_transform_key"],
        )

    @property
    def scaled_intrinsic(self) -> np.ndarray:
        matrix = self.intrinsic.copy()
        matrix[0] *= self.frame_width / self.native_width
        matrix[1] *= self.frame_height / self.native_height
        return matrix

    @property
    def camera_origin_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera)[:3, 3]

    @property
    def ground_to_image_homography(self) -> np.ndarray:
        """Project homogeneous road-plane coordinates ``(x, y, 1)`` to pixels."""
        if self.ground_to_image_override is not None:
            return self.ground_to_image_override
        rotation = self.world_to_camera[:3, :3]
        translation = self.world_to_camera[:3, 3]
        return self.scaled_intrinsic @ np.column_stack((rotation[:, 0], rotation[:, 1], translation))

    @property
    def image_to_ground_homography(self) -> np.ndarray:
        """Perspective rectification from image pixels to the metric road plane."""
        matrix = np.linalg.inv(self.ground_to_image_homography)
        return matrix / matrix[2, 2]

    def image_to_bev_homography(
        self,
        extent: tuple[float, float, float, float],
        width: int,
        height: int,
        padding: int = 0,
    ) -> np.ndarray:
        """Map the camera image directly into a north-up metric BEV canvas."""
        x_min, x_max, y_min, y_max = extent
        sx = (width - 2 * padding) / (x_max - x_min)
        sy = (height - 2 * padding) / (y_max - y_min)
        world_to_bev = np.asarray(
            [[sx, 0.0, padding - sx * x_min], [0.0, -sy, height - padding + sy * y_min], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        return world_to_bev @ self.image_to_ground_homography

    def pixel_to_ground(self, u: float, v: float, ground_z: float = 0.0) -> tuple[float, float]:
        """Intersect an image ray with the metric road plane."""
        if self.ground_to_image_override is not None and abs(ground_z) < 1e-9:
            point = self.image_to_ground_homography @ np.asarray([u, v, 1.0])
            if abs(point[2]) < 1e-9:
                raise ValueError("point projects to infinity")
            return float(point[0] / point[2]), float(point[1] / point[2])
        ray_camera = np.linalg.inv(self.scaled_intrinsic) @ np.asarray([u, v, 1.0])
        rotation = self.world_to_camera[:3, :3]
        ray_world = rotation.T @ ray_camera
        origin = self.camera_origin_world
        if abs(ray_world[2]) < 1e-9:
            raise ValueError("image ray is parallel to the road plane")
        distance = (ground_z - origin[2]) / ray_world[2]
        if distance <= 0:
            raise ValueError("image ray intersects the road plane behind the camera")
        point = origin + distance * ray_world
        return float(point[0]), float(point[1])

    def ground_to_pixel(self, point: tuple[float, float, float]) -> tuple[float, float] | None:
        """Project one metric world point into the processed camera frame."""
        if self.ground_to_image_override is not None and abs(point[2]) < 1e-9:
            pixel = self.ground_to_image_override @ np.asarray([point[0], point[1], 1.0])
            if abs(pixel[2]) < 1e-9:
                return None
            return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])
        homogeneous = np.asarray([*point, 1.0], dtype=float)
        camera_point = self.world_to_camera @ homogeneous
        if camera_point[2] <= 1e-8:
            return None
        pixel = self.scaled_intrinsic @ camera_point[:3]
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])

    def project_vehicle_box(
        self,
        center: tuple[float, float],
        heading_deg: float,
        dimensions: tuple[float, float, float],
    ) -> list[tuple[float, float]]:
        """Project the eight corners of a metric vehicle cuboid."""
        length, width, height = dimensions
        angle = np.deg2rad(heading_deg)
        forward = np.asarray([np.cos(angle), np.sin(angle)])
        left = np.asarray([-np.sin(angle), np.cos(angle)])
        origin = np.asarray(center)
        corners: list[tuple[float, float]] = []
        for z in (0.0, height):
            for longitudinal, lateral in (
                (length / 2, width / 2),
                (length / 2, -width / 2),
                (-length / 2, -width / 2),
                (-length / 2, width / 2),
            ):
                xy = origin + longitudinal * forward + lateral * left
                projected = self.ground_to_pixel((float(xy[0]), float(xy[1]), z))
                if projected is None:
                    return []
                corners.append(projected)
        return corners
