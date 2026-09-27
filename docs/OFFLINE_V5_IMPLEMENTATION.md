# Offline V5 — 多帧几何初始化与可审计恢复

## 本轮原因与实现

原始素材第 79–81 帧，实例掩膜包入前后两辆车。V4 对一次拟合立即初始化，ID 26 以 truck 类别和约 3.28 名义米高度启动；后续分离后的观测被旧位置门控拒绝，错误形状和轨迹继续外推。两辆真实车辆分别保留 ID 24/26，不按平面重叠合并。

- 首次几何发布需要连续 3 帧同类别、高质量独立原始拟合；尺寸相对中位数偏差至多 15%，同时检查运动残差和速度。检测 ID 生命周期独立，待定位对象保留 ID 和原因。
- 未锁定的错误几何在连续拒绝或类别变化后，可由 6 帧独立证据提出新假设；新投影轮廓 IoU 至少 0.8 且比旧假设提升 0.05，经过区域和残差检查才替换。
- 替换写入 `geometry_epoch` 和事件，保留旧类别、尺寸、位置速度与新候选证据；锁定尺寸不自动替换。RTS 按 epoch 分段，避免跨修订平滑掩盖跳变。
- 这些阈值是工程参数，不是统计置信区间。相邻帧证据也不等于独立真实精度标注。
- 未调用外部 LLM 推理或图像生成；重用同源本地 YOLO 检测缓存进行受控对比。没有车道吸附、碰撞挪车或基于重叠删除车辆。

## 240 帧配对核验

| 指标 | V4 | V5 |
|---|---:|---:|
| 发布唯一 ID | 37 | 37 |
| 发布对象—帧 | 4273 | 4283 |
| 有平面定位对象—帧 | 3638 | 3560 |
| 待定位对象—帧 | 635 | 723 |
| 占地框重叠车辆对—帧（面积 >0.1 名义平方米） | 19 | 0 |
| 同一 3303 个已观测且定位样本：投影凸包 IoU 中位数 | 0.81185 | 0.82266 |
| 同上 IoU P10 | 0.60806 | 0.68978 |

定位覆盖净减少 78 条，约 2.1%。共同发布样本中，96 条新增定位、184 条失去定位；发布集合另有 11 条新增、1 条减少。全片 ID 数相同不代表逐帧发布集合相同。完整清单在 `v5-regressions.json`。

ID 24 首次定位从第 79 帧延迟到 85；ID 26 从 81 延迟到 96。两者随后保留至第 239 帧。原 19 次重叠中，15 次对应 V5 尚未同时定位，另外 4 次（96–99）两版本均定位且 V5 消除重叠。因此零重叠部分来自更谨慎发布，不是全部由位置校正产生。

本次 240 帧未触发六帧几何恢复事件；该分支已实现且候选生成/分段有单元测试，但仍需另一个真实触发样例验收。主要实测改善来自初始化门槛。

原 V4 回归原样执行通过：ID 8 第 57–60 帧持续有位置、拒绝量测状态标记正确、离线处理不改在线逐帧 ID 集合、输入哈希一致。

## 重要未解决项

1. 标定仍为假定内参与道路尺度；缺少同场景实测控制点、独立校验点和车辆尺寸真值。位置/车道/尺寸精度均未验收。
2. ID 24/26 的初始尺寸约 2.50×2.01×2.25 和 2.50×2.04×2.37 名义米，明显不符合普通轿车比例；即使轮廓贴合度上升，也不代表恢复了真实体积。后续应检查标定、朝向、轮廓约束退化和先验边界，而不是改图标比例遮盖。
3. 原始背景地面逆投影中，树木、车身等非地面对象存在拉伸和残影；不是精确道路正射地图。
4. 测试仅约 8 秒单场景。没有实测 IDF1、车道准确率、位置误差或跨场景结论。
5. 更谨慎的初始化会延迟平面对象出现。待定位与已定位数量明确区分，不伪造坐标满足数量相等。

## 复现

在 `/home/hcr/workspaces/carom-rebuild` 执行（输出须为空目录，保留旧版）：

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests -q
.venv/bin/python scripts/run_carom_method.py --max-frames 240 --experimental-calibration --detection-cache outputs/carom-lifecycle-v3/detections.jsonl --output outputs/carom-lifecycle-v5-final --video
.venv/bin/python scripts/smooth_carom_offline.py outputs/carom-lifecycle-v5-final outputs/carom-offline-v5-final
.venv/bin/python scripts/verify_carom_method.py outputs/carom-offline-v5-final
.venv/bin/python scripts/check_v4_regressions.py outputs/carom-offline-v5-final
.venv/bin/python scripts/check_v5_regressions.py outputs/carom-offline-v4-final outputs/carom-offline-v5-final
.venv/bin/python scripts/export_method_mobile.py outputs/carom-offline-v5-final
.venv/bin/python scripts/export_bev_detail.py outputs/carom-offline-v5-final
```

主 API 默认切到 `outputs/carom-offline-v5-final`；V4 原产物保留。页面版本以 `/api/carom-method/run-info` 为准，V5 仍明确标记实验、精度未验证。

本轮 94 项后端测试通过（一个既有 Starlette 弃用警告）；`git diff --check` 通过。浏览器实测显示 offline-v5，并核验第 99 帧 ID 24/26 分别有 accepted 量测和独立位置。没有在本轮重复完整的前端构建/长时间交互测试；前端代码本轮未改动。
