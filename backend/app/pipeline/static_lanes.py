"""Static surveyed lane membership. Never changes the estimated pose."""
import math


def lane_membership(x, y, lanes, longitudinal_bounds, heading=None, boundary_margin=.2):
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('nonfinite position')
    result = {'lane_id': None, 'position_was_snapped': False, 'status': 'outside_static_map'}
    if not longitudinal_bounds[0] <= y <= longitudinal_bounds[1]:
        return result
    candidates = []
    for lane in lanes:
        width = float(lane['width_m'])
        if not math.isfinite(width) or width <= 0:
            raise ValueError('invalid lane width')
        offset = x - float(lane['center_lateral_m'])
        if abs(offset) <= width/2 + boundary_margin:
            candidates.append((abs(offset), offset, lane))
    if not candidates:
        return result
    candidates.sort(key=lambda item: item[0])
    _, offset, lane = candidates[0]
    ambiguous = len(candidates) > 1 or abs(offset) > float(lane['width_m'])/2-boundary_margin
    result.update(lane_id=lane['lane_id'] if not ambiguous else None,
                  candidate_lane_ids=[c[2]['lane_id'] for c in candidates],
                  road_id=lane['road_id'], lateral_offset_m=offset,
                  status='boundary_ambiguous' if ambiguous else 'static_map_membership')
    if heading is not None:
        dx, dy = lane['forward_road_xy']
        angle = math.atan2(dy, dx)
        result['direction_difference_rad'] = abs(math.atan2(math.sin(heading-angle), math.cos(heading-angle)))
    return result
