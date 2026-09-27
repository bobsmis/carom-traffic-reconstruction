import unittest
import carla
import numpy as np
from carom_sim.geometry import calibration,make_road,project
from carom_sim.long_tunnel import camera_nodes,LENGTH

class GeometryTests(unittest.TestCase):
    def test_length_and_overlap(self):
        nodes=camera_nodes()
        self.assertEqual(LENGTH,248*5)
        self.assertEqual(len(nodes),10)
        self.assertTrue(all(b['global_offset_m']-a['global_offset_m']==124 for a,b in zip(nodes,nodes[1:])))
        self.assertIn('length="1400"',make_road(1400))
    def test_camera_local_to_shared_ground(self):
        expected=np.array([1.75,195-56])
        for node in camera_nodes()[:2]:
            tf=carla.Transform(carla.Location(*node['location']),carla.Rotation(pitch=node['pitch'],yaw=node['yaw']))
            c=calibration(tf.get_inverse_matrix(),960,540,90,node['origin_x'])
            point=np.array([[1.75,195-node['origin_x'],0.]])
            uv=project(point,c['projection_matrix'])[0]
            ground=np.asarray(c['image_to_ground'])@np.r_[uv,1.]
            global_xy=ground[:2]/ground[2]+[0,node['global_offset_m']]
            self.assertTrue(np.allclose(global_xy,expected,atol=1e-5))

if __name__=='__main__':unittest.main()
