import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[Video_Utilities] gif_preview.js loaded");

function ensurePreview(node) {
    let widget = node.widgets?.find(w => w.name === "gifpreview");
    if (widget) return widget;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.padding = "0";
    container.style.margin = "0";
    container.style.background = "#1a1a1a";
    container.style.borderRadius = "4px";

    const w = node.addDOMWidget("gifpreview", "preview", container, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return container.value; },
        setValue(v) { container.value = v; }
    });
    w.container = container;
    return w;
}

function setMediaSrc(el, filename, dirName) {
    const ext = (filename.split('.').pop() || '').toLowerCase();
    const params = new URLSearchParams({ filename, type: dirName.toLowerCase() });
    const url = api.apiURL('/view?' + params);
    if (ext === 'gif') {
        const img = document.createElement('img');
        img.src = url;
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.display = 'block';
        el.innerHTML = '';
        el.appendChild(img);
    } else {
        const video = document.createElement('video');
        video.controls = true;
        video.loop = true;
        video.muted = true;
        video.src = url;
        video.style.width = '100%';
        video.style.height = 'auto';
        video.style.display = 'block';
        el.innerHTML = '';
        el.appendChild(video);
    }
}

app.registerExtension({
    name: "Ken.VideoUtilities.GIFPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        // 仅接管 Video_To_GIF（更稳健的多条件匹配）
        const name = nodeData?.name || "";
        const comfyClass = nodeData?.comfyClass || nodeType?.comfyClass || "";
        const isTarget = name === "Video_To_GIF" || name.includes("Video_To_GIF") ||
                         name === "VideoToGIFNode" || comfyClass === "VideoToGIFNode";
        if (!isTarget) return;
        console.log("[Video_Utilities] gif_preview attached to", nodeData?.name);

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            try {
                const ui = output?.ui || output?.[0]?.ui || output?.result?.ui;
                const uiVideo = ui?.video;
                if (uiVideo && uiVideo.length >= 2) {
                    const [filename, dirName] = uiVideo;
                    const w = ensurePreview(this);
                    setMediaSrc(w.container, filename, dirName);
                    this.setDirtyCanvas(true, true);
                    console.log("[Video_Utilities] gif_preview render", filename, dirName);
                }
            } catch (e) {
                console.warn("GIF preview error", e);
            }
            return onExecuted?.apply?.(this, arguments);
        }
    }
});


