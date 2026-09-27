# CAROM Studio 1.0 — Windows 本地结果客户端

## 使用
1. 解压完整 CAROM-Studio 文件夹，双击 CAROM-Studio.exe；保留旁边的 _internal 文件夹。
2. 首次启动自动保存两个内置结果组：真实道路 V5（3 路、8 秒）和 CARLA 短样例（4 路、6 秒）。它们是已有算法输出，不是新生成的精度合格演示。
3. “载入 3 / 4 路视频”：填写名称、选择真实/CARLA，为原始、标注、平面各选 MP4，第 4 路可选 SUMO、Unity 或其他结果。保存后复制到素材库。
4. “导出结果包”生成 .carom.zip，包含 3/4 路视频、清单和 SHA-256；“导入结果包”在其他电脑恢复，不依赖原文件路径。
5. 同步播放、暂停、时间轴、倍速统一控制全部实际通道。第 4 路缺失时显示说明而非伪造内容。每组续播位置自动保存。
6. 本机 WebUI：http://127.0.0.1:8767/?library 。CARLA 采集处理仍在 8766，真实视频旧工作台仍在 5173，均需各自后台启动。本客户端不附带 CARLA、CUDA、模型权重，也不代替采集/推理服务。

## 保存位置
视频组：%LOCALAPPDATA%\CAROM Studio\groups
浏览器续播配置：%LOCALAPPDATA%\CAROM Studio\browser
故障日志：%LOCALAPPDATA%\CAROM Studio\startup-error.txt
关闭客户端后视频仍保存。请用导出包备份，不要只备份 exe。导入/导出会临时占用额外磁盘空间。
支持 Windows 10/11 x64，需要 Microsoft Edge WebView2 Runtime；无需单独安装 Python、Node、WSL。只监听本机回环接口，不向公网开放。

## 结果包协议
ZIP 根目录：manifest.json、raw.mp4、annotation.mp4、bev.mp4，可选 extra.mp4。
```
{"schema":"carom-video-group-v1","name":"测试场景","kind":"real","channels":{"raw":{"file":"raw.mp4","label":"原始视频"},"annotation":{"file":"annotation.mp4","label":"算法标注"},"bev":{"file":"bev.mp4","label":"平面投影"}}}
```
kind 为 real 或 carla。可选 note 描述来源和限制。导出时每通道写入 sha256；导入会核对。只接受标准清单和对应文件，拒绝越界路径、多余文件与校验不符。

## 明确的边界
- 推荐 H.264 MP4；MP4 是容器，其他编码是否能解码取决于 WebView2。格式错误显示提示，不做隐式转码。
- 输入必须自行保证同一时间起点。播放器按共同时间轴校正漂移；时长不一致时提示并限制到最短通道。不是帧级同步实验工具，不会自动消除源视频已有时差。
- 回放优化不改变检测、标定或跨摄像头关联精度。导入的所有框、轨迹和 ID 均来自原视频；不读取 CARLA 真值补造结果。
- 目前导入的是已处理文件，不是 RTSP/HLS 实时协议。CARLA 10 个相机可作为 10 组导入，每组含 4 路。
- Windows 包未进行商业代码签名；本机已验证打包启动，其他机器需验证 WebView2 和系统兼容性。

## 开发与测试
源码 desktop/client.py 为标准库本地服务，pywebview 为桌面外壳；React 前端为 frontend/src/Library.tsx。
依赖：pywebview==6.2.1、pyinstaller==6.22.3。
`python -m unittest discover -s desktop -p test_client.py -v`
`python desktop/client.py --serve --port 8767 --data <素材库目录>` 可不启动桌面窗口用于浏览器测试。
`python desktop/import_existing.py --data <素材库目录> --real <demo目录> --carla <已完成run目录>` 将现有完整结果复制入库。
打包前先在 frontend 执行 npm run build && npm run lint，再执行 desktop/build_windows.ps1（按环境修改 Python 路径）。
冻结程序使用官方 pywebview/PyInstaller 工作流：https://pywebview.flowrl.com/guide/freezing
