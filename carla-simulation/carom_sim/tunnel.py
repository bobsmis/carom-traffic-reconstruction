"""Procedural CARLA geometry and scenario controls; never imported by perception."""
import carla


def build_tunnel(world):
    actors=[]
    bp=world.get_blueprint_library().find('static.prop.mesh')
    bp.set_attribute('mesh_path','/Engine/BasicShapes/Cube.Cube')
    bp.set_attribute('scale','8')
    bp.set_attribute('role_name','carom_baseline')
    try:
        for x in range(56,297,8):
            for y,z in [(y,24) for y in (-16,-8,0,8,16)]+[(y,z) for y in (-18,18) for z in (4,12,20)]:
                actors.append(world.spawn_actor(bp,carla.Transform(carla.Location(x=x,y=y,z=z))))
        return actors
    except BaseException:
        for actor in actors: actor.destroy()
        raise


def spawn_entry(world, road, options, rng, tm, lane, lane_changes=True):
    # Entry points are beyond the measurement ROI; never spawn into occupied space.
    wp=road.get_waypoint_xodr(1,lane,55 if lane<0 else 290)
    tf=wp.transform;tf.location.z+=.5
    if any(a.get_location().distance(tf.location)<12 for a in world.get_actors().filter('vehicle.*')):
        return None
    bp=rng.choice(options);bp.set_attribute('role_name','carom_baseline')
    actor=world.try_spawn_actor(bp,tf)
    if actor:
        actor.set_autopilot(True,8050)
        tm.auto_lane_change(actor,lane_changes)
        tm.vehicle_percentage_speed_difference(actor,rng.uniform(5,30))
    return actor
