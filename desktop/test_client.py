import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from client import Library, byte_range, make_server, validate_manifest


def manifest():
    return {'schema':'carom-video-group-v1','name':'Test','kind':'real','channels':{r:{'file':r+'.mp4','label':r} for r in ('raw','annotation','bev')}}


def package(value=None, extra=None):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        z.writestr('manifest.json',json.dumps(value or manifest()))
        for r in ('raw','annotation','bev'): z.writestr(r+'.mp4',b'\0\0\0\x18ftypisom'+b'\0'*64)
        if extra: z.writestr(extra,b'test')
    data.seek(0)
    return data


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.lib=Library(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_roundtrip(self):
        ident=self.lib.import_zip(package())
        rows=self.lib.listing()
        self.assertEqual(rows[0]['id'],ident)
        self.assertEqual(len(rows[0]['channels']['raw']['sha256']),64)
        self.assertEqual(len(Library(self.temp.name).listing()),1)
    def test_four_channels(self):
        m=manifest();m['channels']['extra']={'file':'extra.mp4','label':'SUMO'}
        self.assertEqual(len(validate_manifest(m)['channels']),4)
    def test_two_channels_rejected(self):
        m=manifest();del m['channels']['bev']
        with self.assertRaises(ValueError): validate_manifest(m)
    def test_traversal_rejected(self):
        with self.assertRaises(ValueError): self.lib.import_zip(package(extra='../outside.mp4'))
        self.assertFalse(list(self.lib.groups.iterdir()))
    def test_bad_checksum_rejected_atomically(self):
        m=manifest();m['channels']['raw']['sha256']='0'*64
        with self.assertRaises(ValueError): self.lib.import_zip(package(m))
        self.assertFalse(list(self.lib.groups.iterdir()))
        self.assertFalse(list(self.lib.pending.iterdir()))
    def test_bad_id(self):
        with self.assertRaises(ValueError): self.lib.stage('../outside')
    def test_ranges(self):
        self.assertEqual(byte_range('bytes=3-8',100),(3,8,True))
        self.assertEqual(byte_range('bytes=-10',100),(90,99,True))
        self.assertEqual(byte_range('bytes=90-',100),(90,99,True))
        for value in ['bytes=100-', 'bytes=4-1', 'bytes=-0', 'bytes=1-2,4-5']:
            with self.assertRaises(ValueError): byte_range(value,100)
    def test_http_import_range_export(self):
        server=make_server(self.temp.name,self.temp.name,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            req=urllib.request.Request(base+'/library/import',data=package().read(),headers={'X-CAROM-Client':'1'})
            with urllib.request.urlopen(req) as r: ident=json.load(r)['id']
            req=urllib.request.Request(base+f'/library/media/{ident}/raw.mp4',headers={'Range':'bytes=4-7'})
            with urllib.request.urlopen(req) as r:
                self.assertEqual(r.status,206);self.assertEqual(r.read(),b'ftyp')
                self.assertIn('immutable',r.headers['Cache-Control'])
            with urllib.request.urlopen(base+f'/library/export/{ident}') as r:
                with zipfile.ZipFile(io.BytesIO(r.read())) as z: self.assertEqual(len(z.namelist()),4)
            req=urllib.request.Request(base+'/library/import',data=b'bad',headers={'X-CAROM-Client':'1','Origin':'https://other.example'})
            with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(req)
            self.assertEqual(caught.exception.code,403)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
