"""Loopback-only operator service. Fixed actions, no browser-provided commands."""
import json
import mimetypes
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
REPO='/home/hcr/workspaces/carom-rebuild'
DIST=Path(r'\\wsl.localhost\Ubuntu-24.04\home\hcr\workspaces\carom-rebuild\frontend\dist')
CARLA=Path(r'C:\Users\hcrrc\Downloads\CARLA_0.9.16\CarlaUE4.exe')
lock=threading.Lock()
state={'busy':False,'action':None,'error':None,'run':None,'api_version':3}
for manifest in sorted((ROOT/'runs').glob('*/input/manifest.json')):
    state['run']=manifest.parents[1].name

def invoke(command, log):
    with log.open('a',encoding='utf-8') as out:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)

def job(action, body):
    try:
        log=ROOT/'control.log'
        if action=='launch':
            import carla
            c=carla.Client('127.0.0.1',2000);c.set_timeout(2)
            try:c.get_server_version()
            except RuntimeError:
                subprocess.Popen([str(CARLA),'-windowed','-ResX=1280','-ResY=720'],cwd=CARLA.parent,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        elif action in ('capture','capture-multi'):
            import carla
            c=carla.Client('127.0.0.1',2000);c.set_timeout(10);w=c.get_world()
            # Only remove assets tagged by this project's previous runs.
            for a in w.get_actors():
                if a.attributes.get('role_name') in ('carom_baseline','carom_multicam'):a.destroy()
            (ROOT/'stop.request').unlink(missing_ok=True)
            before=set((ROOT/'runs').iterdir())
            multi=action=='capture-multi';fps=10 if multi else 20
            command=[sys.executable,str(ROOT/('capture_multicam.py' if multi else 'capture_tunnel.py')),'--frames',str(body['seconds']*fps),
                     '--fps',str(fps),'--seed',str(body['seed'])]
            if not multi:command.append('--keep-scene')
            else:
                command+=['--brightness',str(body.get('brightness',70))]
                (ROOT/'scene-controls.json').write_text(json.dumps({'brightness':body.get('brightness',70)}))
            if body['lane_changes']:command.append('--lane-changes')
            if body['collision'] and not multi:command.append('--collision')
            invoke(command,log)
            created=set((ROOT/'runs').iterdir())-before
            if created:state['run']=sorted(created)[-1].name
        elif action=='process-multi':
            run=body['run'];src='/mnt/c/Users/hcrrc/PycharmProjects/PythonProject3/runs/'+run+'/input'
            command=f'cd {REPO} && .venv/bin/python scripts/process_multicam.py --input {src} --name multi-{run}'
            invoke(['wsl.exe','-d','Ubuntu-24.04','--','bash','-lc',command],log)
            state['run']=run
        elif action=='process':
            run=body['run'];name='tunnel-'+run
            src='/mnt/c/Users/hcrrc/PycharmProjects/PythonProject3/runs/'+run+'/input'
            if not (ROOT/'runs'/run/'input/manifest.json').exists():raise ValueError('Selected capture is incomplete')
            # Values are validated identifiers; no arbitrary filesystem/shell inputs.
            frames=json.loads((ROOT/'runs'/run/'input/manifest.json').read_text())['frames']
            if type(frames) is not int or not 1<=frames<=12000:raise ValueError('Invalid manifest frames')
            command=f'cd {REPO} && if [ ! -f configs/{name}.json ]; then .venv/bin/python scripts/import_carla_input.py {src} --name {name}; fi && .venv/bin/python scripts/run_carom_method.py --config configs/{name}.json --output outputs/{name} --max-frames {frames} --video'
            invoke(['wsl.exe','-d','Ubuntu-24.04','--','bash','-lc',command],log)
            import imageio_ffmpeg
            ff=imageio_ffmpeg.get_ffmpeg_exe()
            output=Path(r'\\wsl.localhost\Ubuntu-24.04\home\hcr\workspaces\carom-rebuild\outputs')/name
            media=ROOT/'runs'/run/'web';media.mkdir(exist_ok=True)
            for kind,source,crop in [('raw',ROOT/'runs'/run/'input/rgb.mp4',None),
                ('annotation',output/'method-preview.mp4','crop=960:540:960:44'),
                ('bev',output/'method-preview.mp4','crop=1540:440:0:630')]:
                cmd=[ff,'-hide_banner','-loglevel','error','-y','-i',str(source)]
                if crop:cmd+=['-vf',crop]
                invoke(cmd+['-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(media/(kind+'.mp4'))],log)
            state['run']=run
    except Exception as exc:state['error']=str(exc)
    finally:
        state['busy']=False

def start(action, body):
    with lock:
        if state['busy']:raise ValueError('Another task is running')
        state.update(busy=True,action=action,error=None)
        threading.Thread(target=job,args=(action,body),daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def send_json(self,data,status=200):
        payload=json.dumps(data).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)))
        self.end_headers();self.wfile.write(payload)
    def do_POST(self):
        if self.headers.get('Host') not in ('127.0.0.1:8766','localhost:8766'):
            return self.send_json({'error':'Host rejected'},403)
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8766','http://localhost:8766'):
            return self.send_json({'error':'Origin rejected'},403)
        if self.headers.get('Content-Type')!='application/json':return self.send_json({'error':'JSON required'},415)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<4096:raise ValueError('invalid request size')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ValueError('JSON object required')
            action=self.path.removeprefix('/sim-api/')
            if action=='brightness':
                value=body.get('brightness')
                if type(value) not in (int,float) or not 0<=value<=100:raise ValueError('brightness: 0..100')
                temp=ROOT/'scene-controls.tmp';temp.write_text(json.dumps({'brightness':value}));temp.replace(ROOT/'scene-controls.json')
            elif action=='stop':
                if state['busy'] and state['action'] in ('capture','capture-multi'):(ROOT/'stop.request').touch()
                else:raise ValueError('No active capture')
            elif action in ('launch','capture','capture-multi','process','process-multi'):
                if action in ('capture','capture-multi'):
                    limit=180 if action=='capture-multi' else 120
                    if type(body.get('seconds')) is not int or not 3<=body['seconds']<=limit:raise ValueError(f'seconds: 3..{limit}')
                    if type(body.get('seed')) is not int or not 0<=body['seed']<2**31:raise ValueError('invalid seed')
                    if any(type(body.get(k)) is not bool for k in ('lane_changes','collision')):raise ValueError('boolean flags required')
                    if action=='capture-multi' and (type(body.get('brightness',70)) not in (int,float) or not 0<=body.get('brightness',70)<=100):raise ValueError('brightness: 0..100')
                if action in ('process','process-multi') and not re.fullmatch(r'\d{8}-\d{6}',str(body.get('run',''))):raise ValueError('invalid run')
                start(action,body)
            else:return self.send_json({'error':'Unknown action'},404)
            self.send_json({'accepted':True},202)
        except (ValueError,KeyError) as exc:self.send_json({'error':str(exc)},400)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/sim-api/status':
            runs=[]
            for folder in sorted((ROOT/'runs').iterdir(),reverse=True):
                if (folder/'run.json').exists():
                    try:info=json.loads((folder/'run.json').read_text(encoding='utf-8'))
                    except (json.JSONDecodeError,UnicodeError):
                        info={'completed':False,'frames_requested':0}
                    runs.append({'id':folder.name,'completed':info['completed'],'frames':info['frames_requested'],
                                 'schema':info.get('schema','single'),
                                 'processed':(folder/'web/bev.mp4').exists() or (folder/'web/complete.json').exists()})
            return self.send_json({**state,'runs':runs,'log':(ROOT/'control.log').read_text(encoding='utf-8',errors='replace')[-3000:] if (ROOT/'control.log').exists() else ''})
        if path=='/sim-api/events':
            run=state['run'];p=ROOT/'runs'/run/'truth/events.jsonl' if run else None
            return self.send_json([json.loads(s) for s in p.read_text().splitlines()] if p and p.exists() else [])
        if path=='/sim-api/live.jpg':
            images=sorted((ROOT/'runs').glob('*/diagnostics/live.jpg'))
            if not images:return self.send_json({'error':'No live frame yet'},404)
            return self.file(images[-1])
        cam=re.fullmatch(r'/sim-api/(cam(?:0[1-9]|10))\.jpg',path)
        if cam:
            images=sorted((ROOT/'runs').glob('*/diagnostics/'+cam[1]+'.jpg'))
            return self.file(images[-1]) if images else self.send_json({'error':'No camera frame'},404)
        multi=re.fullmatch(r'/sim-media/(\d{8}-\d{6})/(cam(?:0[1-9]|10)/)?(raw|annotation|bev|sumo|global)\.mp4',path)
        if multi:return self.file(ROOT/'runs'/multi[1]/'web'/(multi[2] or '')/(multi[3]+'.mp4'))
        data=re.fullmatch(r'/sim-data/(\d{8}-\d{6})/(global|cameras)\.json',path)
        if data:return self.file(ROOT/'runs'/data[1]/('web' if data[2]=='global' else 'input')/(data[2]+'.json'))
        match=re.fullmatch(r'/sim-media/(\d{8}-\d{6})/(raw|annotation|bev)\.mp4',path)
        if match:return self.file(ROOT/'runs'/match[1]/'web'/(match[2]+'.mp4'))
        target=(DIST/path.lstrip('/')).resolve() if path!='/' else DIST/'index.html'
        if not target.is_relative_to(DIST.resolve()):return self.send_json({'error':'Invalid path'},403)
        return self.file(target)
    def file(self,path):
        if not path.is_file():return self.send_json({'error':'Not found'},404)
        total=path.stat().st_size;start=0;end=total-1;status=200
        r=self.headers.get('Range')
        if r:
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',r)
            if not match:return self.send_json({'error':'Invalid range'},416)
            start=int(match[1]);end=min(int(match[2]) if match[2] else end,end);status=206
            if start>end:return self.send_json({'error':'Range outside file'},416)
        self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    data=stream.read(min(65536,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)
        except (ConnectionResetError,BrokenPipeError):pass

if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
