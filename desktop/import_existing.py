"""Copy completed project results into the portable library, never rerun inference."""
import argparse
import json
import shutil
import uuid
from pathlib import Path
from client import Library


def register(lib, folder, name, kind, files, note):
    ident=uuid.uuid5(uuid.NAMESPACE_URL,str(folder.resolve())+'|'+name).hex
    if (lib.groups/ident).exists(): return ident
    stage=lib.stage(ident)
    channels={}
    for role,(source,label) in files.items():
        shutil.copyfile(folder/source,stage/(role+'.mp4'))
        channels[role]={'file':role+'.mp4','label':label}
    lib.commit(ident,{'schema':'carom-video-group-v1','name':name,'kind':kind,'channels':channels,'note':note})
    return ident


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--real');p.add_argument('--carla');a=p.parse_args();lib=Library(a.data)
    if a.real:
        folder=Path(a.real)
        register(lib,folder,'真实道路 · V5 已处理样例','real',{'raw':('original.mp4','原始视频'),'annotation':('annotated.mp4','算法标注'),'bev':('bev.mp4','平面投影')},'现有 V5 离线输出，标定精度未经验证。')
    if a.carla:
        run=Path(a.carla)
        if not (run/'web/complete.json').is_file(): raise ValueError('CARLA 结果尚未处理完成')
        for folder in sorted((run/'web').glob('cam[0-9][0-9]')):
            register(lib,folder,f'CARLA · {run.name} · {folder.name}','carla',{'raw':('raw.mp4','节点原始 RGB'),'annotation':('annotation.mp4','全局 ID 标注'),'bev':('bev.mp4','局部平面投影'),'extra':('sumo.mp4','SUMO 视觉轨迹回放')},'仅复制已处理输出，不使用仿真真值补充算法结果。')
    print(json.dumps({'groups':len(lib.listing()),'data':str(lib.root)},ensure_ascii=False))

if __name__=='__main__':main()
