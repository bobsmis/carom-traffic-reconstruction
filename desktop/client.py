"""Portable, loopback-only library for preprocessed CAROM video groups."""
import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROLES = ('raw', 'annotation', 'bev', 'extra')
MAX_BYTES = 20 * 1024**3


def validate_manifest(value):
    if not isinstance(value, dict) or value.get('schema') != 'carom-video-group-v1':
        raise ValueError('需要 carom-video-group-v1 结果包')
    if value.get('kind') not in ('real', 'carla'):
        raise ValueError('素材类型应为 real 或 carla')
    if 'note' in value and (not isinstance(value['note'], str) or len(value['note']) > 4000):
        raise ValueError('素材说明格式错误')
    channels = value.get('channels')
    if not isinstance(channels, dict) or set(channels) not in (set(ROLES[:3]), set(ROLES)):
        raise ValueError('每组必须包含原始、标注、平面三路，可选第四路')
    if not isinstance(value.get('name'), str) or not value['name'].strip() or len(value['name']) > 160:
        raise ValueError('素材名称为空或过长')
    for role, item in channels.items():
        if not isinstance(item, dict) or item.get('file') != role + '.mp4':
            raise ValueError('通道文件名必须为 raw/annotation/bev/extra.mp4')
        if not isinstance(item.get('label'), str) or len(item['label']) > 100:
            raise ValueError('通道标题格式错误')
    return value


def byte_range(header, size):
    if not header:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()):
        raise ValueError('Invalid range')
    left, right = match.groups()
    if not left:
        length = int(right)
        if length <= 0: raise ValueError('Invalid suffix')
        start, end = max(0, size-length), size-1
    else:
        start, end = int(left), min(int(right), size-1) if right else size-1
    if start > end or start >= size: raise ValueError('Range outside file')
    return start, end, True


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.groups = self.root / 'groups'
        self.pending = self.root / 'pending'
        for path in (self.groups, self.pending): path.mkdir(parents=True, exist_ok=True)

    def listing(self):
        results = []
        for path in sorted(self.groups.glob('*/manifest.json')):
            try:
                item = validate_manifest(json.loads(path.read_text(encoding='utf-8')))
                results.append({**item, 'id': path.parent.name})
            except (ValueError, OSError):
                continue
        return results

    def stage(self, ident):
        if not re.fullmatch('[a-f0-9]{32}', ident): raise ValueError('Invalid group identifier')
        path = self.pending / ident
        path.mkdir(exist_ok=True)
        return path

    def commit(self, ident, manifest):
        validate_manifest(manifest)
        directory = self.stage(ident)
        for channel in manifest['channels'].values():
            path = directory / channel['file']
            if not path.is_file() or path.stat().st_size < 12: raise ValueError('视频缺失或为空')
            with path.open('rb') as source:
                if source.read(12)[4:8] != b'ftyp': raise ValueError('仅接受 MP4 容器的视频')
            hasher = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''): hasher.update(block)
            digest = hasher.hexdigest()
            if channel.get('sha256') and channel['sha256'] != digest: raise ValueError('视频校验失败')
            channel['sha256'] = digest
        (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        directory.rename(self.groups / ident)
        return ident

    def import_zip(self, archive):
        ident = uuid.uuid4().hex
        directory = self.stage(ident)
        try:
            with zipfile.ZipFile(archive) as package:
                if len(package.infolist()) > 6: raise ValueError('结果包包含多余文件')
                info = package.getinfo('manifest.json')
                if info.file_size > 65536: raise ValueError('Manifest too large')
                manifest = validate_manifest(json.loads(package.read('manifest.json')))
                allowed = {'manifest.json'} | {c['file'] for c in manifest['channels'].values()}
                if len(package.namelist()) != len(allowed) or set(package.namelist()) != allowed:
                    raise ValueError('结果包文件清单不匹配')
                if sum(i.file_size for i in package.infolist()) > MAX_BYTES: raise ValueError('结果包过大')
                for name in allowed - {'manifest.json'}:
                    with package.open(name) as src, (directory/name).open('wb') as dst:
                        shutil.copyfileobj(src, dst, 1024*1024)
            return self.commit(ident, manifest)
        except Exception:
            # Only our UUID staging directory; never touches user's input archive.
            shutil.rmtree(directory)
            raise

    def seed(self, seed_root):
        for path in Path(seed_root).glob('*/manifest.json'):
            ident = uuid.uuid5(uuid.NAMESPACE_URL, 'carom-builtin-' + path.parent.name).hex
            if (self.groups / ident).exists(): continue
            target = self.stage(ident)
            manifest = validate_manifest(json.loads(path.read_text(encoding='utf-8')))
            for item in manifest['channels'].values(): shutil.copyfile(path.parent/item['file'], target/item['file'])
            self.commit(ident, manifest)


class Handler(BaseHTTPRequestHandler):
    server_version = 'CAROMLibrary/1.0'

    def log_message(self, *_): pass

    def json_reply(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def trusted(self, mutate=False):
        host = self.headers.get('Host', '')
        if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://' + host,): return False
        if self.headers.get('Sec-Fetch-Site') == 'cross-site': return False
        return not mutate or self.headers.get('X-CAROM-Client') == '1'

    def do_POST(self):
        if not self.trusted(True): return self.json_reply({'error': 'Origin rejected'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BYTES: raise ValueError('文件为空或超过 20 GB')
            route = urlsplit(self.path).path
            match = re.fullmatch('/library/upload/([a-f0-9]{32})/(raw|annotation|bev|extra)', route)
            if match:
                ident, role = match.groups()
                file = self.server.library.stage(ident) / (role + '.mp4')
                self.save_body(file, length)
                return self.json_reply({'ok': True})
            if route == '/library/import':
                file = self.server.library.pending / (uuid.uuid4().hex + '.zip')
                try:
                    self.save_body(file, length)
                    ident = self.server.library.import_zip(file)
                finally: file.unlink(missing_ok=True)
                return self.json_reply({'id': ident})
            if route == '/library/commit':
                if length > 65536: raise ValueError('Manifest too large')
                body = json.loads(self.rfile.read(length))
                ident = self.server.library.commit(body['id'], body['manifest'])
                return self.json_reply({'id': ident})
            return self.json_reply({'error': 'Not found'}, 404)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            return self.json_reply({'error': str(error)}, 400)

    def save_body(self, file, length):
        self.connection.settimeout(120)
        with file.open('wb') as stream:
            remaining = length
            while remaining:
                data = self.rfile.read(min(1024*1024, remaining))
                if not data: raise ValueError('上传中断，请重新导入')
                stream.write(data)
                remaining -= len(data)

    def do_HEAD(self): self.do_GET(head=True)

    def do_GET(self, head=False):
        if not self.trusted(): return self.json_reply({'error': 'Origin rejected'}, 403)
        route = unquote(urlsplit(self.path).path)
        if route == '/library/groups': return self.json_reply({'groups': self.server.library.listing()})
        match = re.fullmatch('/library/media/([a-f0-9]{32})/(raw|annotation|bev|extra).mp4', route)
        if match:
            ident, role = match.groups()
            return self.serve_file(self.server.library.groups/ident/(role+'.mp4'), head, immutable=True)
        match = re.fullmatch('/library/export/([a-f0-9]{32})', route)
        if match:
            ident = match[1]
            directory = self.server.library.groups / ident
            if not (directory/'manifest.json').exists(): return self.send_error(404)
            # Unique temp name makes parallel exports safe, original media untouched.
            archive = self.server.library.pending / (uuid.uuid4().hex + '.carom.zip')
            try:
                manifest = validate_manifest(json.loads((directory/'manifest.json').read_text(encoding='utf-8')))
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as package:
                    for name in ['manifest.json'] + [c['file'] for c in manifest['channels'].values()]:
                        package.write(directory/name, name)
                return self.serve_file(archive, head, download='CAROM-' + ident[:8] + '.carom.zip')
            finally: archive.unlink(missing_ok=True)
        target = (self.server.dist / (route.lstrip('/') or 'index.html')).resolve()
        if not target.is_relative_to(self.server.dist.resolve()): return self.send_error(403)
        return self.serve_file(target, head)

    def serve_file(self, path, head, immutable=False, download=None):
        if not path.is_file(): return self.send_error(404)
        size = path.stat().st_size
        try: start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0'); self.end_headers(); return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'private, max-age=31536000, immutable' if immutable else 'no-cache')
        if download: self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        if partial: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head: return
        try:
            with path.open('rb') as stream:
                stream.seek(start); remaining = end-start+1
                while remaining:
                    data = stream.read(min(1024*1024, remaining))
                    if not data: break
                    self.wfile.write(data); remaining -= len(data)
        except (ConnectionError, OSError): pass  # Browser seek can cancel a range request.


def make_server(root, dist, port=8767):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.library = Library(root)
    server.dist = Path(dist)
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--data')
    args = parser.parse_args()
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
    root = Path(args.data) if args.data else Path(os.environ.get('LOCALAPPDATA', Path.home()))/'CAROM Studio'
    dist = bundle/'web'
    if not dist.exists(): dist = Path(__file__).parent.parent/'frontend/dist'
    server = make_server(root, dist, args.port)
    server.library.seed(bundle/'examples')
    if args.serve: server.serve_forever(); return
    import webview
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webview.settings['ALLOW_DOWNLOADS'] = True
    webview.create_window('CAROM Studio · 真实道路与仿真结果工作台', f'http://127.0.0.1:{args.port}/?library', width=1440, height=960, min_size=(960,640))
    try: webview.start(gui='edgechromium', private_mode=False, storage_path=str(root/'browser'))
    finally: server.shutdown(); server.server_close()


if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback
        log = Path(os.environ.get('LOCALAPPDATA', Path.home()))/'CAROM Studio'
        log.mkdir(parents=True, exist_ok=True)
        (log/'startup-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if not getattr(sys, 'frozen', False): raise
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, '启动失败，可能端口已占用或缺少 WebView2。详情：'+str(log/'startup-error.txt'), 'CAROM Studio', 0x10)


