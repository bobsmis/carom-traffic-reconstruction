import http.client
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from control_server import Handler

class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()
    def post(self,path,data,origin='http://127.0.0.1:8766',host='127.0.0.1:8766'):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        c.request('POST',path,data,{'Content-Type':'application/json','Origin':origin,'Host':host})
        r=c.getresponse();status=r.status;r.read();c.close();return status
    def test_foreign_origin(self):self.assertEqual(self.post('/sim-api/launch','{}',origin='https://other.example'),403)
    def test_foreign_host(self):self.assertEqual(self.post('/sim-api/launch','{}',host='other.example'),403)
    def test_arbitrary_command(self):self.assertEqual(self.post('/sim-api/exec','{}'),404)
    def test_traversal_run(self):self.assertEqual(self.post('/sim-api/process','{"run":"../../outside"}'),400)
    def test_unbounded_capture(self):self.assertEqual(self.post('/sim-api/capture','{"seconds":999999,"seed":1,"lane_changes":true,"collision":true}'),400)
    def test_invalid_brightness(self):self.assertEqual(self.post('/sim-api/brightness','{"brightness":101}'),400)
    def test_json_array_rejected(self):self.assertEqual(self.post('/sim-api/brightness','[]'),400)
    def test_three_minute_multi_allowed(self):
        with patch('control_server.start') as start:
            self.assertEqual(self.post('/sim-api/capture-multi','{"seconds":180,"seed":84,"lane_changes":true,"collision":false}'),202)
            start.assert_called_once()
    def test_more_than_three_minutes_rejected(self):
        self.assertEqual(self.post('/sim-api/capture-multi','{"seconds":181,"seed":84,"lane_changes":true,"collision":false}'),400)

if __name__=='__main__':unittest.main()
