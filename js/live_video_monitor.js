import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 基础样式（与参考项目一致的外观）
const style = document.createElement('style');
style.textContent = `
  .livevideo_preview {
    position: relative;
    width: 100%;
    overflow: hidden;
    background: #2a2a2a;
    border-radius: 4px;
    margin: 0;
    padding: 0;
    display: block !important;
  }
  .livevideo_preview video {
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;
    background: #1a1a1a;
    margin: 0;
    padding: 0;
  }
`;
document.head.appendChild(style);

function fitHeight(node, aspectRatio) {
  if (!node || !node.setSize || !aspectRatio) return;
  const nodeWidth = node.size?.[0] ?? 400;
  const videoHeight = Math.max(1, (nodeWidth - 40) / aspectRatio);
  // 控件与标题的保留空间，取适中值，避免遮挡
  const reserved = 230;
  node.setSize([nodeWidth, videoHeight + reserved]);
  node?.graph?.setDirtyCanvas(true);
}

function clearPreview(node) {
  if (!node || !Array.isArray(node.widgets)) return;
  const controlNames = ["refresh_interval", "auto_refresh", "min_file_age"]; // 保留控制部件
  for (let i = node.widgets.length - 1; i >= 0; i--) {
    const w = node.widgets[i];
    if (!w || controlNames.includes(w.name)) continue;
    try { w.parentEl?.remove?.(); } catch {}
    node.widgets.splice(i, 1);
  }
}

function preview(node, filename, type) {
  clearPreview(node);
  const element = document.createElement("div");
  element.setAttribute("data-livevideo-node-id", node.id);

  const previewWidget = node.addDOMWidget("livevideopreview", "preview", element, {
    serialize: false,
    hideOnZoom: false,
    getValue() { return element.value; },
    setValue(v) { element.value = v; },
  });

  const parentEl = document.createElement("div");
  parentEl.className = "livevideo_preview";
  parentEl.style.width = "100%";
  element.appendChild(parentEl);

  const videoEl = document.createElement("video");
  videoEl.controls = true;
  videoEl.loop = false;
  videoEl.muted = false;
  videoEl.autoplay = false; // 禁用自动播放，由用户控制
  videoEl.preload = "metadata";
  parentEl.appendChild(videoEl);

  videoEl.addEventListener("loadedmetadata", () => {
    const ar = (videoEl.videoWidth || 16) / Math.max(1, videoEl.videoHeight || 9);
    previewWidget.aspectRatio = ar;
    fitHeight(node, ar);
  });
  videoEl.addEventListener("error", () => {
    parentEl.hidden = true;
    node?.graph?.setDirtyCanvas(true);
  });

  const params = new URLSearchParams({
    filename,
    type: (type?.toLowerCase?.() === "input") ? "input" : "output",
  });

  // 智能选择端点：通过 API 检测视频编码，MPEG-4 视频使用转码
  (async () => {
    try {
      // 调用编码检测 API
      const detectUrl = api.apiURL('/video_utilities/detect_codec?' + new URLSearchParams({
        filename: filename,
        type: (type?.toLowerCase?.() === "input") ? "input" : "output"
      }));

      const response = await fetch(detectUrl);
      const data = await response.json();

      const needsTranscode = data.needs_transcode || false;
      const codec = data.codec || 'unknown';

      const endpoint = needsTranscode ? '/video_utilities/viewvideo' : '/view';
      videoEl.src = api.apiURL(endpoint + '?' + params.toString());
      // 强制禁用自动播放
      videoEl.autoplay = false;

      console.log("🎬 Live_Video_Monitor: File:", filename);
      console.log("🎬 Live_Video_Monitor: Codec:", codec);
      console.log("🎬 Live_Video_Monitor: Needs transcode:", needsTranscode);
    } catch (error) {
      console.warn("⚠️ Live_Video_Monitor: Codec detection failed, using /view:", error);
      videoEl.src = api.apiURL('/view?' + params.toString());
      // 强制禁用自动播放
      videoEl.autoplay = false;
    }
  })();
}

app.registerExtension({
  name: "Ken-Chen_VideoUtilities.LiveVideoMonitor",
  async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
    // 同时兼容类名与显示名
    const isTarget = nodeData?.name === "VideoUtilitiesLiveVideoMonitor" || nodeData?.name === "Live_Video_Monitor";
    if (!isTarget) return;

    // 自动刷新执行，仿照参考项目
    nodeType.prototype.onNodeCreated = function () {
      this.__live_timer = null;
      const start = () => {
        if (this.__live_timer) {
          clearInterval(this.__live_timer);
          this.__live_timer = null;
        }
        const refreshInterval = this.widgets?.find(w=>w.name==="refresh_interval")?.value ?? 2;
        const autoRefresh = this.widgets?.find(w=>w.name==="auto_refresh")?.value !== false;
        const minFileAge = this.widgets?.find(w=>w.name==="min_file_age")?.value ?? 1;
        if (!autoRefresh) return;

        const execute = async () => {
          try {
            const prompt = {
              [this.id]: {
                inputs: {
                  refresh_interval: refreshInterval,
                  auto_refresh: autoRefresh,
                  min_file_age: minFileAge,
                },
                class_type: "VideoUtilitiesLiveVideoMonitor",
              },
            };
            await api.queuePrompt(0, prompt);
          } catch (err) { console.log("LiveVideoMonitor auto exec error", err); }
        };

        setTimeout(execute, 300);
        this.__live_timer = setInterval(execute, Math.max(1, refreshInterval) * 1000);
      };

      // 延迟启动，保证widgets就绪
      setTimeout(start, 800);
    };

    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
      try {
        if (data && Array.isArray(data.video) && data.video.length >= 2) {
          preview(this, data.video[0], data.video[1]);
        }
      } catch (e) { console.log(e); }
      if (origExecuted) return origExecuted.apply(this, arguments);
    };

    const origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try { if (this.__live_timer) clearInterval(this.__live_timer); } catch {}
      this.__live_timer = null;
      if (origRemoved) return origRemoved.apply(this, arguments);
    };
  }
});


