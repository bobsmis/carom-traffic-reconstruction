import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Release from "./Release";
import Workbench from "./Workbench";
import Simulation from "./Simulation";
import MultiSimulation from "./MultiSimulation";

import Library from "./Library";
const params = new URLSearchParams(window.location.search);
if (params.has('legacy') || params.has('evidence')) void import('./styles.css');

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {params.has('library') ? <Library /> : params.has('multi') ? <MultiSimulation /> : params.has('sim') || location.port === '8766' ? <Simulation /> : params.has("legacy") ? <><aside role="note">历史静态演示，含预置状态；不是当前离线处理结果。<a href="/">返回工作台</a></aside><App /></> : params.has('evidence') ? <Release /> : <Workbench />}
  </StrictMode>,
);
