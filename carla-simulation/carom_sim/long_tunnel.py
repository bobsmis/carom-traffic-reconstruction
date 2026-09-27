"""Native textured tunnel segments, shared static camera registry."""
import math
import carla

ROLE='carom_multicam'
START=56.
LENGTH=1240.
ROAD_LENGTH=1400

def camera_nodes():
    return [{'id':f'cam{i+1:02}', 'station_m':i*124.,'global_offset_m':i*124.,
             'origin_x':START+i*124.,'location':[START+i*124.-10.,-10.,12.],
             'pitch':-22.,'yaw':8.,'roi_m':[0.,150.]} for i in range(10)]

def build(world):
    """Overlapping solid voussoirs form a closed, faceted semicircular shell."""
    bp=world.get_blueprint_library().find('static.prop.mesh')
    bp.set_attribute('mesh_path','/Engine/BasicShapes/Cube.Cube')
    bp.set_attribute('scale','8');bp.set_attribute('role_name',ROLE)
    actors=[]
    try:
        steps=math.ceil(LENGTH/8);pitch=LENGTH/steps
        for j in range(steps):
            x=START+(j+.5)*pitch
            for k in range(10):
                angle=(k+.5)*math.pi/10
                rotation=carla.Rotation(roll=-math.degrees(angle))
                pos=carla.Location(x=x,y=22*math.cos(angle),z=22*math.sin(angle))
                actors.append(world.spawn_actor(bp,carla.Transform(pos,rotation)))
        return actors
    except BaseException:
        for a in actors:
            if a.is_alive:a.destroy()
        raise
