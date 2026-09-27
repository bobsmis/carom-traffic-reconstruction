# CAROM / CARLA 基准场景 v1

这是算法验证用的原生 CARLA 场景，不是 WebUI，也不是最终城市美术场景。

## 已完成

- CARLA 0.9.16：500 m 平直道路，双向各 4 车道，车道宽 3.5 m。
- 24 辆物理仿真车辆，Traffic Manager 驱动；基准阶段关闭变道。
- 固定路侧 RGB 相机：1280×720，水平 FOV 90°，20 FPS。
- 同步采集 RGB、实例标签、车辆状态及相机内外参，以仿真帧号配对。
- 本地 CAROM 重建算法只导入 input/ 中的 RGB 和静态标定/道路数据。
- 当前完成数据：runs/20260926-230110，160 帧（8 秒）。
- 摄像机、车辆和道路为真实 CARLA 渲染/仿真；道路标线使用 CARLA debug 静态线条，尚非正式材质。无建筑背景，非照片级场景。

## Python 与启动

PyCharm 项目原来选择 Python 3.13，该环境未装 CARLA。将项目解释器选择为：

`C:\Users\hcrrc\miniconda3\envs\carla\python.exe`

在本目录 PowerShell 执行：

```powershell
$py = 'C:\Users\hcrrc\miniconda3\envs\carla\python.exe'
& $py audit_capture.py runs/20260926-230110
# 下面清理仅删除该次运行记录且 role_name 匹配的车辆：
& $py cleanup_owned.py runs/20260926-230110
# 创建新道路前须为空场景；程序遇到其他车辆/传感器会停止，保留它们。
& $py capture_baseline.py --frames 240 --fps 20 --keep-scene
```

服务默认 localhost:2000，Traffic Manager 使用 8050。每次生成道路会替换当前地图；当前运行前已检查世界没有车辆和传感器。采集后恢复异步设置，保留的车辆冻结供检查；不自动恢复原 Town10HD_Opt 地图。

## 数据隔离

|目录|用途|
|---|---|
|input/|算法输入：RGB、帧索引、精确模拟相机参数、静态道路模型|
|truth/|评估专用：车辆真实尺寸/位置/方向、实例标签|
|diagnostics/|明确标为 ground truth 的渲染检查，不算算法检测结果|
|implementation/|采集代码快照|

这是文件与程序路径分离，不是操作系统权限沙箱。算法没有读取动态车辆真值。精确相机参数是允许的标定输入，不代表检测和重建已经精确。

道路坐标 X=CARLA Y，Y=CARLA X−100，Z=CARLA Z；右手坐标。相机外参转换至 OpenCV 坐标。没有地理坐标标定，所有测量是本地米制。

## 与现有算法连接

WSL 仓库 `/home/hcr/workspaces/carom-rebuild` 已增加 `scripts/import_carla_input.py` 和精确模拟外参适配。当前场景配置为 `configs/carla-eight-lane-baseline.json`，算法输出为 `outputs/carla-eight-lane-baseline/`。导入器仅接收 input/；重建仍使用本地 YOLO 分割与现有几何/跟踪，不拿真值充当检测。

## 验收边界 / 下一步

采集审计覆盖帧数、时间同步、数据摘要、车辆清单和车道数，并非算法精度评价。远处漏检、初始化等待、ID 中断仍会出现，不能把场景搭建完成等同于问题已解决。

后续先验证实例标签与 actor_id 的对应规则，再评估位置误差、车道准确率、遮挡恢复和 ID 切换；不可直接假设实例标签颜色就是 CARLA actor_id。相机视锥内也不等于未遮挡可见。

随后加入单车/多车、遮挡、变道、停车和光照变化独立场景。真实视频泛化、正式道路材质、原生四窗口客户端及 Unity 接入仍是独立后续工作。
