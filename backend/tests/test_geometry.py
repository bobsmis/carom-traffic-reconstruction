import numpy as np

from app.geometry import project_points, solve_homography, vehicle_footprint


def test_homography_maps_calibration_corners() -> None:
    image = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
    ground = [(-5.0, 40.0), (5.0, 40.0), (5.0, 0.0), (-5.0, 0.0)]
    matrix = solve_homography(image, ground)
    projected = project_points(matrix, image)
    assert np.allclose(projected, ground)


def test_homography_maps_center() -> None:
    image = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
    ground = [(-5.0, 40.0), (5.0, 40.0), (5.0, 0.0), (-5.0, 0.0)]
    matrix = solve_homography(image, ground)
    assert np.allclose(project_points(matrix, [(5.0, 10.0)]), [(0.0, 20.0)])


def test_homography_accepts_redundant_road_controls() -> None:
    image = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0), (5.0, 0.0), (5.0, 20.0)]
    ground = [(-5.0, 40.0), (5.0, 40.0), (5.0, 0.0), (-5.0, 0.0), (0.0, 40.0), (0.0, 0.0)]
    matrix = solve_homography(image, ground)
    assert np.allclose(project_points(matrix, image), ground)


def test_vehicle_footprint_preserves_metric_dimensions() -> None:
    footprint = np.asarray(vehicle_footprint((0.0, 0.0), 4.6, 1.8, 0.0))
    assert np.isclose(np.linalg.norm(footprint[0] - footprint[1]), 1.8)
    assert np.isclose(np.linalg.norm(footprint[1] - footprint[2]), 4.6)
