# CAROM Traffic Reconstruction — public source and data

固定单目相机交通重建、CARLA 多相机采集与统一三/四路视频回放客户端。

## 数据与客户端下载
见 [v1.0.0-data Release](https://github.com/bobsmis/carom-traffic-reconstruction/releases/tag/v1.0.0-data)。
完整批次 `20260927-102144`：10 相机，1727 帧/相机，10 FPS，172.7 秒。
- `carla-20260927-102144-input.zip`：原始 RGB、标定、时间戳及场景配置。
- `carla-20260927-102144-results.zip`：40 路处理后视频、全局轨迹及辅助结果。
- `carla-20260927-102144-truth.zip`：独立评估真值和实例掩码；不得作为感知推理输入。
- `CAROM-Studio-Windows-x64.zip`：Windows 免安装回放客户端。
解压三个数据包至同一父目录，即恢复 `20260927-102144/` 完整目录。请核对 `SHA256SUMS.txt`。

## 重要边界
这是研究原型，不宣称完成高精度或全场景复现。入口生成导致中部相机有空场等待期；既有标定、检测和跨镜关联限制仍保留。
便携客户端只回放已处理数据；完整处理需要另行配置模型、CARLA、SUMO、Python/Node 和本机路径。
素材/图标包含用户提供及第三方来源，发布不等于授予其新的许可；复用前核对各自权利与许可。CARLA 引擎本体、模型权重和本机环境不随包分发。

---

# CAROM Rebuild — Offline V1.0 RC1

当前交付为 V1 工程验收候选版，使用用户提供的无检测标注车流素材。
入口默认读取 `outputs/release-v1`；旧演示保留在 `?legacy`，两者数据独立。
详细问题分析、复现命令和未达标项见 [V1 验收说明](docs/V1_RELEASE.md)。

**精度声明：** 当前采用人工道路对应点和标称 3.5 米车道尺度，尚无独立实测定位/速度真值。
车辆尺寸为分割接地点与车型先验的估计，拟合框不代表实测三维体积。
所有道路几何、车辆观测、模型修正、重叠冲突均可追溯；运行阶段无 LLM 调用。

以下为早期 V0.2 演示文档，仅描述旧演示的数据与接口，并非 V1 精度证明。

固定单目相机交通流空间重建。当前版本按照 CAROM 论文链路完成前三个可验证阶段：原始帧、车辆实例/体积框、米制道路平面重建；最终 3D 场景渲染暂缓。

## 当前功能

- 原始固定相机画面，保持无覆盖层，便于逐像素核验
- 每辆车的实例轮廓、稳定跟踪编号、类别尺寸先验及透视体积框
- 8 个道路控制点的超定平面单应性，将车辆底面中心映射到米制道路坐标
- 道路边界、车道分界、停止线、导流区和车道编号的显式语义模型
- 按车辆真实长宽生成的朝向 footprint，不再用统一大小的块状图标
- 每个对象的历史轨迹、速度向量、定位不确定度、角速度和转向状态
- 点击原图任意位置，后端即时返回地面坐标并在两幅图中同步标记
- 前后端分离：React/Vite 前端，FastAPI 后端
- 完整离线运行；当前帧来自 CAROM 官方演示缩略图的无标注相机区域

## 启动

```bash
cd /home/hcr/workspaces/carom-rebuild
./scripts/start.sh
```

打开 <http://localhost:5173>。接口文档位于 <http://localhost:8000/docs>。

## 验收

```bash
./scripts/check.sh
```

验收包含 TypeScript 构建、ESLint、几何投影单元测试和 API 测试。

## 导出三阶段结果图

```bash
python scripts/export_three_stage.py /mnt/c/Users/hcrrc/Desktop/大创/CAROM_three_stage_result.png
```

导出图包含原始相机帧、实例掩膜/车辆体积框、米制道路平面和车辆状态，不包含最终 3D 场景。

## 数据接口

- `GET /api/scene`：场景、标定、检测及地面坐标
- `GET /api/frame`：原始固定相机帧
- `POST /api/project`：像素点到米制地面坐标
- `GET /api/health`：离线服务状态

`backend/data/scene.json` 是离线场景数据契约。后续接入 YOLO-seg、光流跟踪器和逐帧状态滤波时保持该输出结构，前端无需重写。

## 与论文方法的对应关系

| 当前阶段 | CAROM 方法 | 本版本实现 |
| --- | --- | --- |
| 01 | 固定单目视频输入 | 原始 RGB 帧 |
| 02 | 实例分割、跟踪、车型识别、朝向/3D 框估计 | 实例 polygon、稳定 ID、类别尺寸先验、透视体积框、朝向来源 |
| 03 | 图像到世界平面变换、地图坐标车辆状态 | 多点 homography、语义道路、米制 footprint、轨迹与不确定度 |
| 04 | 3D 场景可视化 | 暂缓 |

当前附带帧只有单帧素材，因此轨迹和转向字段用于验证完整数据契约；接入连续视频后由光流/跟踪器逐帧更新。道路图是基于可见道路特征建立的局部米制模型，接入现场测量点或正射底图后可直接替换控制点与语义几何。

## 真实视频流水线

项目现已增加一条独立于演示 `scene.json` 的本地视频流水线：

```text
公开固定相机视频
  → YOLO 实例分割
  → ByteTrack 稳定 ID
  → mask 底部接地点
  → 官方内参/外参射线与道路平面求交
  → 常速度 Kalman 状态滤波
  → 速度、朝向、角速度和转向状态
  → 车型尺寸先验投影体积框
  → 轨迹重采样和层次聚类
  → 自动路线中心线与宽度
  → 三阶段视频、逐帧 JSONL、路线 JSON 和审计 JSON
```

运行过程不调用语言模型，也不读取 `backend/data/scene.json` 中的演示车辆字段。每个输出值在 `frames.jsonl` 中记录算法来源，`audit.json` 固定记录：

- `llm_runtime_calls: 0`
- `manual_vehicle_annotations: 0`
- 输入视频、配置和模型权重 SHA-256
- 使用的相机标定文件
- 实际处理帧数、检测数、稳定 ID 和路线数

### 素材与标定

默认演示场景已切换为 UA-DETRAC 的干净固定相机片段。源视频像素中没有数据集发布方烧录的框、编号或轨迹；数据集标注与运行时完全分离。该场景使用道路标线/停止线控制点拟合像素到米制平面的单应矩阵，并保留控制点与来源审计。

TUMTraf 配置继续作为高精度标定基准，使用 Dev Kit 发布的 `s040_camera_basler_north_50mm`：

- 相机内参矩阵与畸变参数
- 相机到道路坐标系外参
- 1920×1200 原始相机尺寸
- 固定基础设施相机的 GPS/UTM 元数据

TUMTraf 公开预览本身带有发布方可视化，因此不再作为默认展示素材。需要验证官方内外参时仍可显式选择该配置。

### 安装推理依赖

```bash
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
pip install -r backend/requirements.txt
pip install lap==0.5.13
```

### 下载并运行

```bash
python scripts/download_public_sample.py \
  --config configs/ua_detrac_clean_intersection.json
python scripts/run_video_pipeline.py \
  --config configs/ua_detrac_clean_intersection.json \
  --output outputs/ua-detrac-clean-intersection

# 导出论文风格的三阶段验证板（省略最终 3D 场景）
python scripts/export_pipeline_validation.py \
  --config configs/ua_detrac_clean_intersection.json \
  --input outputs/ua-detrac-clean-intersection \
  /mnt/c/Users/hcrrc/Desktop/大创/CAROM_pipeline_validation.png
```

输出：

- `three_stage.mp4`：光流、立体框、实例掩膜、注册地图四路同步视频
- `result_frame.jpg`：自动选择的代表帧
- `frames.jsonl`：逐帧对象、坐标、速度、朝向和字段来源
- `routes.json`：由真实轨迹聚类产生的路线
- `audit.json`：输入与运行时审计信息
- `CAROM_pipeline_validation.png`：原始帧、实例/体积框、米制道路平面的三阶段验证板

### 用户天桥素材测试场景

`configs/user_bridge_traffic.json` 使用用户提供的 1920×1080 双向道路固定机位视频。场景采用八条名义 3.5 m 车道建立道路平面单应性，并注册到用户提供的八车道俯视道路图。`car`、`bus`、`truck` 在地图中使用透明背景俯视车辆素材；短时漏检对象继续沿所属车道运动并降低图标透明度。

```bash
python scripts/run_video_pipeline.py \
  --config configs/user_bridge_traffic.json \
  --output outputs/user-bridge-traffic

python scripts/export_pipeline_validation.py \
  --config configs/user_bridge_traffic.json \
  --input outputs/user-bridge-traffic \
  /mnt/c/Users/hcrrc/Desktop/大创/CAROM_user_material_validation.png
```

该配置中的绝对米制比例来自名义车道宽度，轨迹、车道归属和相对位置可直接检查；需要将速度作为测量真值时，应把四个道路控制点替换为现场测绘坐标。

新增接口：

- `GET /api/pipeline/audit`
- `GET /api/pipeline/routes`
- `GET /api/pipeline/result-frame`
- `GET /api/pipeline/video`

### 多场景复用

每个摄像机使用一个 JSON 配置，指定视频、裁剪区域及标定方式。已有官方内外参时使用完整针孔模型；仅有实测道路控制点时使用 `planar_calibration` 的像素—地面对应关系。检测、跟踪、状态估计、路线聚类及导出代码保持不变。

### V0.3 透视校正与轨迹约束

- 对公开视频烧录的高饱和度可视化线条先执行固定 HSV 掩膜与 OpenCV 修复，再交给本地检测器；该步骤只处理像素，不读取发布方类别、编号或轨迹。
- ByteTrack 的新轨阈值与检测阈值统一，低置信度远处车辆也会进入编号流程；同帧高 IoU 重复框会被抑制。
- 相机内外参直接构成道路平面单应矩阵，时间中值背景经逆透视映射生成米制 BEV 道路纹理。
- 轨迹先去除突刺并平滑，再按行驶方向和横向位置聚类；车道中心线使用共享道路曲线拟合，避免对不同长度轨迹直接求平均造成交叉。
- 每个车辆同时保留原始米制落地点和车道约束位置 `lane_ground`，并记录横向残差 `lateral_error_m`，便于精度审计。
- 图像侧框体使用实例包络的切线式紧凑框；平面侧使用车型尺寸先验的米制 footprint。

### 全局编号与短时丢失管理

检测器/ByteTrack 的编号只作为单帧关联线索，工程另外维护会话级单调递增的 `track_id`。对象短时漏检后进入 `coasting` 状态，使用 Kalman 常速度模型继续输出位置、速度、朝向和 footprint；重新检测到的对象通过预测米制位置执行匈牙利匹配，即使底层编号发生变化也恢复原全局编号。默认 `max_lost_frames=12`，超过阈值后轨迹进入 retired 状态；同一会话内不复用旧编号。逐帧数据提供：

- `detector_track_id`：底层 ByteTrack 临时编号
- `track_id`：会话级全局编号
- `tracking_status`：`observed` 或 `predicted`
- `lost_frames`：从最后一次真实观测开始累计的漏检帧数
- `prediction_only`：该帧是否由运动模型补全

### 航拍道路底图与相机视域

`road_map` 配置使用世界米制控制点和底图像素控制点建立单应变换。相机画面的边界射线与道路平面求交后形成地面视锥，再与底图上的可行驶区域相交：相机当前覆盖范围保持高亮并绘制青色边界，覆盖范围以外叠加暗色遮罩。显示方向设为逆时针旋转，使视锥顶点和相机标记位于平面图右侧。

底图中的 `lane_centerlines_px` 是地图坐标系内的真实车道中心线。系统先完成像素射线到米制道路平面的透视校正，再拟合并扣除所有可靠轨迹共有的道路曲率；随后依据每个全局轨迹的全生命周期横向残差进行加权聚类，并固定分配到一条地图车道。原始 `ground` 坐标仍完整保留用于误差审计。这样同一弯曲车道上处于不同远近位置的车辆仍会落入同一地图车道，并输出 `map_lane_id`、`map_lane_residual_m`、`map_position_px` 和 `map_lateral_error_px`。

### CAROM 四路同步渲染

离线视频按照同一个逐帧对象记录同步渲染四个画面：二维框与 Lucas–Kanade 稀疏光流、编号与拟合立体框、实例掩膜与全局编号、注册道路地图与车辆状态。四个画面共享完全相同的 `track_id` 集合，不分别执行检测，因此每个相机框都对应一个检测框和一个地图对象。地图对象使用带车窗和车头方向的俯视汽车图标；漏检期间沿常速度状态继续运动，恢复观测时使用有限修正量的渐进融合以抑制闪烁和位置跳变。

## 素材说明

参考实现来自 [duolu/CAROM](https://github.com/duolu/CAROM)。第一版演示帧截取自其 README 链接的官方演示视频缩略图，仅用于本地复现与界面验证。
# CAROM 2021 方法替换分支

新增 [20 px/m 米制俯视详情与可缩放查看页](docs/BEV_DISPLAY_SCALE.md)：在既有结果上运行 `scripts/export_bev_detail.py outputs/carom-method-wide-stable`，不修改车辆坐标。

新增独立论文方法实现，说明和复现命令见 [方法对齐与验收边界](docs/CAROM_METHOD_ALIGNMENT.md)。
入口为 `scripts/run_carom_method.py`，最新扩区与固定尺寸版本输出为 `outputs/carom-method-wide-stable`，API 前缀为 `/api/carom-method`。
有效区域的参数、排除原因和复现方法见 [有效测量区说明](docs/MEASUREMENT_REGION.md)，尺寸策略见 [刚性尺寸说明](docs/VEHICLE_SIZE_STABILITY.md)。`outputs/carom-method-v2` 和 `outputs/carom-method-roi` 分别保留无 ROI 与 50m ROI 的对照结果。
旧 V1 保留为对照；当前用户素材的四点假设标定仅供实验，尚未通过真实精度验收。
