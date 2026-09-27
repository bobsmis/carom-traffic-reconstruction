"""Encode the computed preview for browsers; never regenerate scene content."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


def export(output):
    source=output/'method-preview.mp4'
    if not source.is_file():
        raise FileNotFoundError(source)
    target=output/'mobile.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                    '-i',str(source),'-vf','scale=1280:720','-c:v','libx264','-preset','fast',
                    '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    def info(path):
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'video validation failed: {path}')
        result={'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':cap.get(cv2.CAP_PROP_FPS),
                'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        return result
    original,encoded=info(source),info(target)
    assert original['frames']==encoded['frames'] and abs(original['fps']-encoded['fps'])<.01
    (output/'mobile-export.json').write_text(json.dumps({'original':original,'mobile':encoded},indent=2))
    print(json.dumps(encoded))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output',type=Path)
    export(parser.parse_args().output)
