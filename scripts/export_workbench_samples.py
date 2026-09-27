"""Export real V5 evidence for the standalone frontend; no synthesized traffic."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'outputs/carom-offline-v5-final'
    out = ROOT / 'frontend/public/demo'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'scene-config.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    crop = config['crop']
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    def encode(video, target, filters):
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(video),
                        '-frames:v', str(audit['frames']), '-vf', filters,
                        '-an', '-c:v', 'libx264', '-crf', '21', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(out / target)], check=True)
    encode(ROOT / config['video_path'], 'original.mp4',
           f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},scale=1280:-2")
    encode(source / 'method-preview.mp4', 'annotated.mp4', 'crop=960:540:960:0,scale=960:540')
    shutil.copy2(source / 'bev-detail.mp4', out / 'bev.mp4')
    files = {}
    for name in ('original.mp4', 'annotated.mp4', 'bev.mp4'):
        cap = cv2.VideoCapture(str(out / name))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert frames == audit['frames'], (name, frames)
        files[name] = {'frames': frames, 'fps': fps, 'sha256': hashlib.sha256((out / name).read_bytes()).hexdigest()}
    manifest = {'source_run': source.name, 'source_sha256': audit['source_sha256'],
                'annotation_crop_xywh': [960, 0, 960, 540], 'unity': 'not_connected',
                'accuracy_validated': False, 'files': files}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
