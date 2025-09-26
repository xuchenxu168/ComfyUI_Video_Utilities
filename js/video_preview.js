import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'

console.log("🎬 Video Preview JavaScript 文件已加载");

// Add CSS styles to ensure proper video display
const style = document.createElement('style');
style.textContent = `
    .video_preview {
        position: relative;
        width: 100%;
        overflow: hidden;
        background: #2a2a2a;
        border-radius: 4px;
        margin: 0;
        padding: 0 0 10px 0; /* 添加底部内边距 */
        display: block !important;
        max-height: 80vh; /* 限制最大高度为视口高度的80% */
    }
    
    .video_preview video {
        width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        background: #1a1a1a;
        margin: 0;
        padding: 0;
        max-height: 100%; /* 确保视频不超出容器 */
    }
    
    .video_preview.hidden {
        display: none !important;
    }
    
    /* Ensure video elements don't overlap */
    [data-node-id] video {
        position: relative;
        z-index: 1;
    }
    
    /* Clear any floating elements */
    .video_preview::after {
        content: '';
        display: table;
        clear: both;
    }
`;
document.head.appendChild(style);

function fitHeight(node) {
    if (!node || !node.setSize || !node.size) {
        console.log("fitHeight - node、setSize或size未定义:", node);
        return;
    }
    try {
        node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
        node?.graph?.setDirtyCanvas(true);
    } catch (error) {
        console.error("fitHeight - 错误:", error);
    }
}

function clearPreviousVideo(node) {
    if (!node || !node.widgets || !Array.isArray(node.widgets)) {
        console.log("clearPreviousVideo - node、widgets未定义或不是数组:", node);
        return;
    }
    
    try {
        // Remove all existing video preview widgets
        while (node.widgets.length > 0) {
            let widget = node.widgets.pop();
            if (widget && widget.parentEl) {
                try {
                    widget.parentEl.remove();
                } catch (error) {
                    console.log("Error removing widget parent:", error);
                }
            }
        }
        
        // Clear any remaining video elements that might be orphaned
        if (node.id) {
            const existingVideos = document.querySelectorAll(`[data-node-id="${node.id}"]`);
            existingVideos.forEach(video => {
                try {
                    video.remove();
                } catch (error) {
                    console.log("Error removing orphaned video:", error);
                }
            });
        }
    } catch (error) {
        console.error("clearPreviousVideo - 错误:", error);
    }
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        // The intended way of appending options is returning a list of extra options,
        // but this isn't used in widgetInputs.js and would require
        // less generalization of chainCallback
        let optNew = []
        try {
            const previewWidget = this.widgets.find((w) => w.name === "videopreview");

            let url = null
            if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
                //Use full quality video
                //url = api.apiURL('/view?' + new URLSearchParams(previewWidget.value.params));
                url = previewWidget.videoEl.src
            }
            if (url) {
                optNew.push(
                    {
                        content: "Open preview",
                        callback: () => {
                            window.open(url, "_blank")
                        },
                    },
                    {
                        content: "Save preview",
                        callback: () => {
                            const a = document.createElement("a");
                            a.href = url;
                            a.setAttribute("download", new URLSearchParams(previewWidget.value.params).get("filename"));
                            document.body.append(a);
                            a.click();
                            requestAnimationFrame(() => a.remove());
                        },
                    }
                );
            }
            if(options.length > 0 && options[0] != null && optNew.length > 0) {
                optNew.push(null);
            }
            options.unshift(...optNew);
            
        } catch (error) {
            console.log(error);
        }
        
    });
}

function previewVideo(node,file,type){
    console.log("previewVideo 函数被调用 - node:", node, "file:", file, "type:", type);
    
    // Clear previous video content completely
    clearPreviousVideo(node);
    
    var element = document.createElement("div");
    element.setAttribute("data-node-id", node.id);
    const previewNode = node;
    
    var previewWidget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(v) {
            element.value = v;
        },
    });
    
    previewWidget.aspectRatio = null;
    
    previewWidget.computeSize = function(width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
            if (!(height > 0)) {
                height = 0;
            }
            this.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];//no loaded src, widget should not display
    }
    
    previewWidget.value = {hidden: false, paused: false, params: {}}
    
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "video_preview";
    previewWidget.parentEl.style['width'] = "100%"
    element.appendChild(previewWidget.parentEl);
    
    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = true;
    previewWidget.videoEl.loop = false;
    previewWidget.videoEl.muted = false;
    previewWidget.videoEl.style['width'] = "100%"
    previewWidget.videoEl.setAttribute("data-node-id", node.id);
    previewWidget.videoEl.setAttribute("preload", "metadata");
    
    // Clear any existing source to prevent ghosting
    previewWidget.videoEl.src = "";
    previewWidget.videoEl.load();
    
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
        fitHeight(previewNode);
    });
    previewWidget.videoEl.addEventListener("error", () => {
        //TODO: consider a way to properly notify the user why a preview isn't shown.
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
    });

    // 处理Type参数 - 将目录名转换为ComfyUI期望的类型
    let fileType = "output"; // 默认
    if (type && type.toLowerCase() === "input") {
        fileType = "input";
    } else if (type && type.toLowerCase() === "output") {
        fileType = "output";
    }
    
    let params =  {
        "filename": file,
        "type": fileType,
    }
    
    // 调试信息
    console.log("Preview Video - file:", file, "type:", type, "fileType:", fileType);
    console.log("Preview Video - params:", params);
    
    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    previewWidget.videoEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;
    
    let target_width = 256;
    if (element.style?.width) {
        //overscale to allow scrolling. Endpoint won't return higher than native
        target_width = element.style.width.slice(0,-2)*2;
    }
    
    if (!params.force_size || params.force_size.includes("?") || params.force_size == "Disabled") {
        params.force_size = target_width+"x?";
    } else {
        let size = params.force_size.split("x");
        let ar = parseInt(size[0])/parseInt(size[1]);
        params.force_size = target_width+"x"+(target_width/ar);
    }
    
    let videoUrl = api.apiURL('/view?' + new URLSearchParams(params));
    console.log("Preview Video - 视频URL:", videoUrl);
    
    previewWidget.videoEl.src = videoUrl;
    previewWidget.videoEl.hidden = false;
    previewWidget.parentEl.appendChild(previewWidget.videoEl)
    
    console.log("Preview Video - 视频元素已添加到DOM");
    console.log("Preview Video - 视频元素src:", previewWidget.videoEl.src);
    console.log("Preview Video - 视频元素hidden:", previewWidget.videoEl.hidden);
    console.log("Preview Video - 父元素hidden:", previewWidget.parentEl.hidden);
    
    // Store cleanup function for when node is destroyed
    if (!node._cleanupFunctions) {
        node._cleanupFunctions = [];
    }
    
    node._cleanupFunctions.push(() => {
        if (previewWidget.videoEl) {
            previewWidget.videoEl.pause();
            previewWidget.videoEl.src = "";
            previewWidget.videoEl.load();
        }
        if (previewWidget.parentEl) {
            previewWidget.parentEl.remove();
        }
    });
}

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.VideoPreviewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("VideoPreviewer - 注册节点:", nodeData?.name);
        if (nodeData?.name == "VideoPreviewNode") {
            console.log("VideoPreviewer - 找到 VideoPreviewNode 节点，添加 onExecuted 方法");
            nodeType.prototype.onExecuted = function (data) {
                console.log("VideoPreviewNode onExecuted - 完整数据:", data);
                console.log("VideoPreviewNode onExecuted - this:", this);
                
                if (data && data.video && Array.isArray(data.video) && data.video.length >= 2) {
                    console.log("VideoPreviewNode onExecuted - data.video[0]:", data.video[0]);
                    console.log("VideoPreviewNode onExecuted - data.video[1]:", data.video[1]);
                    previewVideo(this, data.video[0], data.video[1]);
                } else {
                    console.error("VideoPreviewNode - 数据格式错误:", data);
                }
            }
            
            // Add cleanup when node is removed
            const originalOnRemoved = nodeData.onRemoved;
            nodeData.onRemoved = function() {
                if (this._cleanupFunctions) {
                    this._cleanupFunctions.forEach(cleanup => {
                        try {
                            cleanup();
                        } catch (error) {
                            console.log("Error during cleanup:", error);
                        }
                    });
                    this._cleanupFunctions = [];
                }
                if (originalOnRemoved) {
                    originalOnRemoved.call(this);
                }
            };
        }
    }
});
