# CAROM 四视图前端工作台

## 范围
React 19 + TypeScript + Vite，复用现有 lucide-react 图标库；本轮没有改变算法、标定、后端接口或部署网络权限。
默认 `/` 为工作台；`/#online`、`/#offline`、`/#incidents`、`/#replay` 对应四页；`/?evidence` 保留 V5 算法核验，`/?legacy` 保留旧演示。

界面依照用户指定 Codex 浅色工作区：白色主区、灰蓝色右导航、细边框与四宫格。用户要求的四宫格优先于非对称内容布局；整体主区/右栏非对称。复用项目 CSS 工程方式，不引入另一个样式框架；字体 Segoe UI / Microsoft YaHei，图标来自 Lucide。

## 已实现
- 原始、标注、平面、Unity 四个窗口；单窗口放大/还原；视频等比显示。
- Online/Offline 分别保存四路地址，回放页共用 Offline 配置。支持浏览器原生可播放的 HTTP(S) 视频及站内相对路径。输入校验、加载失败和空输入状态可见。
- 离线共用播放、暂停、时间定位、回退一秒、0.5/1/2 倍速；首个已配置通道主时钟，播放中时间差超过 0.2 秒时追齐。仅适用于共同零点的素材，不承诺帧级同步或不同起始时间自动校准。
- 事件页手动记录、搜索、状态筛选、处理/重新打开、二次确认删除、JSON 导出。记录包含浏览器时间、当前播放位置、来源模式；存储于本机 localStorage。
- 桌面右栏、移动端顶部导航与纵向视频。无外部字体/图片依赖。

## 后续接口边界
- 原始 HTML video 不是通用实时协议播放器。RTSP 需要服务端转流；WebRTC 需要信令/连接管理；HLS 需要平台支持或专用适配器。当前未接入这些服务。
- Unity 为预留视频通道，非已运行的 Unity 场景；当前没有 Unity 画面。
- 当前事件是人工记录，不是事故识别算法，不连接事故数据库。回放目前是当前配置片段，不包含历史录像检索、批量目录或后端归档。
- localStorage 非跨设备存储；勿在视频 URL 中放账号、密码或密钥。远程视频是否可播还取决于编码、访问权限及浏览器网络策略。
- 更换离线视频源后，已有事件的时间只保留记录值，不自动绑定到新视频。接入归档服务时应增加 immutable recording_id、source_timestamp、frame_index。

## 样例来源
`frontend/public/demo` 三段 240 帧视频来自 V5 真实本地结果，非生成式图像或伪造检测：原始视频裁剪缩放、已处理画面右上窗口裁剪、已有等比例 BEV 视频。`manifest.json` 记录哈希、帧数、帧率和来源。

复现导出：
```bash
cd /home/hcr/workspaces/carom-rebuild
.venv/bin/python scripts/export_workbench_samples.py
cd frontend
export PATH=/home/hcr/.local/bin:/usr/bin:/bin
npm run build
npm run lint
npm run dev -- --host 127.0.0.1
```
Vite 默认端口 5173。工作台及样例可独立运行；旧核验页仍依赖本地 FastAPI 的 8000 端口。没有发布公网。

## 本轮验证
- TypeScript/Vite 构建、ESLint 零警告检查。
- 浏览器 Offline 三路加载，定位到 3.6 秒后三路 currentTime 均为 3.6；播放按钮状态切换。
- Online 初始 0/4，配置本地测试视频后为 1/4；RTSP 输入显示校验提示；完成测试后清空回 0/4。
- 事件页新增明确标注的临时测试记录，标记已处理，再以页面内二次确认删除；不保留虚构事故。
- 回放页加载共用素材，定位到 3.6 秒后三路一致；导航回 Offline 正常。
- 390px 窄屏检查为单列，无横向内容溢出；结束后恢复桌面尺寸。
- 最终标签页控制台没有 error/warn。
- 原生 confirm 在测试浏览器引起工具阻塞，已更换为页面内二次确认；旧测试标签无交付标记。
- 尚未验证外部实时协议、Unity、长期连续运行及多源不同时间戳同步。
