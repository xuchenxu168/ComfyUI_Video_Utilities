import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'

console.log("=".repeat(80));
console.log("🎬🎬🎬 VIDEO PREVIEW JAVASCRIPT FILE LOADED - VERSION 2.0 🎬🎬🎬");
console.log("=".repeat(80));

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
        // 仅移除我们添加的 DOM 预览小部件（name === 'videopreview' 或带 parentEl 的 DOMWidget）
        for (let i = node.widgets.length - 1; i >= 0; i--) {
            const widget = node.widgets[i];
            const isOurDomWidget = widget && (widget.name === "videopreview" || widget.parentEl);
            if (!isOurDomWidget) continue;
            try { widget.parentEl?.remove?.(); } catch {}
            node.widgets.splice(i, 1);
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

function previewVideo(node,file,subfolder){
    console.log("previewVideo 函数被调用 - node:", node, "file:", file, "subfolder:", subfolder);

    // 检查是否有编码警告（来自 VideoPreviewNode）
    const hasCodecWarning = node._codecWarning;
    const videoPath = node._videoPath;

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
    
    const isGif = typeof file === 'string' && file.toLowerCase().endsWith('.gif');
    if (isGif) {
        // 使用 <img> 预览 GIF
        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%";
        previewWidget.imgEl.style['height'] = "auto";
        previewWidget.imgEl.setAttribute("data-node-id", node.id);
        previewWidget.imgEl.addEventListener("load", () => {
            const w = previewWidget.imgEl.naturalWidth || 1;
            const h = previewWidget.imgEl.naturalHeight || 1;
            previewWidget.aspectRatio = w / h;
            fitHeight(previewNode);
        });
        previewWidget.imgEl.addEventListener("error", () => {
            previewWidget.parentEl.hidden = true;
            fitHeight(previewNode);
        });
    } else {
        // 使用 <video> 预览视频
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
            console.log("Video metadata loaded - dimensions:", previewWidget.videoEl.videoWidth, "x", previewWidget.videoEl.videoHeight);
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(previewNode);
        });

        previewWidget.videoEl.addEventListener("error", (e) => {
            console.error("Video loading error:", e);
            console.error("Video error details:", previewWidget.videoEl.error);

            // 尝试添加更多的视频格式支持
            if (previewWidget.videoEl.error && previewWidget.videoEl.error.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED) {
                console.warn("Video format not supported by browser, trying alternative approach");

                // 检查是否有来自 VideoPreviewNode 的编码警告
                const codecWarningFromNode = node._codecWarning;
                const isTopazVideo = codecWarningFromNode === 'topaz_mpeg4' || file.toLowerCase().includes('topaz');
                const isMpeg4Video = codecWarningFromNode === 'mpeg4' || codecWarningFromNode === 'topaz_mpeg4';

                // 创建一个详细的提示信息
                const errorDiv = document.createElement("div");
                errorDiv.style.cssText = `
                    padding: 15px;
                    background: linear-gradient(135deg, #2d3748, #4a5568);
                    color: #fff;
                    text-align: center;
                    border-radius: 8px;
                    font-size: 13px;
                    border: 1px solid #718096;
                    margin: 5px;
                `;

                let errorMessage = "🎬 Video Preview Not Available\n\n";
                if (isTopazVideo) {
                    errorMessage += "⚠️ Topaz Video AI processed video detected\n";
                    errorMessage += "This video uses MPEG-4 part 2 encoding which has limited browser support.\n\n";
                    errorMessage += "💡 Solutions:\n";
                    errorMessage += "• Video will work normally in ComfyUI workflows\n";
                    errorMessage += "• For preview, consider converting to H.264 format\n";
                    errorMessage += "• Use VHS Load Video nodes for better compatibility";
                } else if (isMpeg4Video) {
                    errorMessage += "⚠️ MPEG-4 part 2 encoding detected\n";
                    errorMessage += "This encoding has limited browser support.\n\n";
                    errorMessage += "💡 Solutions:\n";
                    errorMessage += "• Video will work normally in ComfyUI workflows\n";
                    errorMessage += "• For preview, consider converting to H.264 format";
                } else {
                    errorMessage += "Video format not supported in browser preview.\n";
                    errorMessage += "File will still work in ComfyUI workflows.";
                }
                errorMessage += `\n\n📁 File: ${file}`;

                errorDiv.innerHTML = errorMessage.replace(/\n/g, '<br>');

                // 清除视频元素并显示错误信息
                if (previewWidget.videoEl.parentNode) {
                    previewWidget.videoEl.parentNode.removeChild(previewWidget.videoEl);
                }
                previewWidget.parentEl.appendChild(errorDiv);

                // 设置一个合适的高度
                previewWidget.computeSize = function (width) {
                    return [width, isTopazVideo ? 180 : (isMpeg4Video ? 160 : 120)];
                };
                fitHeight(previewNode);
                return;
            }

            previewWidget.parentEl.hidden = true;
            fitHeight(previewNode);
        });

        // 添加更多事件监听器来调试
        previewWidget.videoEl.addEventListener("loadstart", () => {
            console.log("Video load started");
        });

        previewWidget.videoEl.addEventListener("canplay", () => {
            console.log("Video can start playing");
        });

        previewWidget.videoEl.addEventListener("canplaythrough", () => {
            console.log("Video can play through without buffering");
        });
    }

    // 处理 subfolder 参数
    // subfolder 可能是实际的子文件夹名（如 "sora_videos"），也可能是类型（如 "input"/"output"）
    let fileType = "output"; // 默认
    let actualSubfolder = "";

    if (subfolder) {
        // 如果 subfolder 是 "input" 或 "output"，则作为 type 使用
        if (subfolder.toLowerCase() === "input") {
            fileType = "input";
        } else if (subfolder.toLowerCase() === "output") {
            fileType = "output";
        } else {
            // 否则作为实际的子文件夹名
            actualSubfolder = subfolder;
        }
    }

    let params =  {
        "filename": file,
        "type": fileType,
        "subfolder": actualSubfolder,
    }

    // 调试信息
    console.log("Preview Video - file:", file, "subfolder:", subfolder, "fileType:", fileType, "actualSubfolder:", actualSubfolder);
    console.log("Preview Video - params:", params);
    
    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    if (!isGif && previewWidget.videoEl) {
        // 禁用自动播放，由用户控制
        previewWidget.videoEl.autoplay = false;
    }
    
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
    
    // 使用转码端点
    let mediaUrl;
    if (isGif) {
        mediaUrl = api.apiURL('/view?' + new URLSearchParams(params));
    } else {
        // 强制使用转码端点
        mediaUrl = api.apiURL('/video_utilities/viewvideo?' + new URLSearchParams(params));
    }

    if (isGif) {
        previewWidget.imgEl.src = mediaUrl;
        previewWidget.imgEl.hidden = false;
        previewWidget.parentEl.appendChild(previewWidget.imgEl);
    } else {
        previewWidget.videoEl.src = mediaUrl;
        previewWidget.videoEl.hidden = false;
        previewWidget.parentEl.appendChild(previewWidget.videoEl);

        // 强制加载视频
        previewWidget.videoEl.load();
    }

    console.log("Preview Media - 已添加到DOM");
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
        if (previewWidget.imgEl) {
            previewWidget.imgEl.src = "";
        }
        if (previewWidget.parentEl) {
            previewWidget.parentEl.remove();
        }
    });
}

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.VideoPreviewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("🔍 VideoPreviewer - 注册节点:", nodeData?.name);
        if (nodeData?.name == "VideoPreviewNode" || nodeData?.name == "Video_To_GIF" || nodeData?.name == "VideoToGIFNode" || nodeData?.name == "Preview_GIF" || nodeData?.name == "PreviewGIFNode") {
            console.log("✅ VideoPreviewer - 找到 VideoPreviewNode 节点，添加 onExecuted 方法");

            // 保存原始的 onExecuted（如果存在）
            const originalOnExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (data) {
                console.log("🎬🎬🎬 VideoPreviewNode onExecuted 被调用！");
                console.log("📦 完整数据:", JSON.stringify(data, null, 2));
                console.log("📦 数据类型:", typeof data);
                console.log("📦 this.id:", this.id);
                console.log("📦 this.title:", this.title);

                // 调用原始的 onExecuted（如果存在）
                if (originalOnExecuted) {
                    try {
                        originalOnExecuted.call(this, data);
                    } catch (e) {
                        console.error("❌ 原始 onExecuted 调用失败:", e);
                    }
                }

                // 检查是否有编码警告信息
                let codecWarning = null;
                let videoPath = null;
                if (data && data.ui) {
                    codecWarning = data.ui.codec_warning;
                    videoPath = data.ui.video_path;
                    console.log("📦 检测到 ui 数据:", data.ui);
                }

                // 兼容 Video_To_GIF：data 可能为 { ui: { video:[name, dir] }, result: (...) }
                let videoTuple = null;
                if (data && data.video && Array.isArray(data.video) && data.video.length >= 2) {
                    videoTuple = data.video;
                    console.log("✅ 从 data.video 解析到:", videoTuple);
                } else if (data && data.ui && Array.isArray(data.ui.video)) {
                    videoTuple = data.ui.video;
                    console.log("✅ 从 data.ui.video 解析到:", videoTuple);
                } else if (typeof data === 'string') {
                    try {
                        const full = data;
                        const name = full.split(/[/\\]/).pop();
                        const lower = full.toLowerCase();
                        const dir = lower.includes('/output/') || lower.includes('\\output\\') ? 'output' : (lower.includes('/input/') || lower.includes('\\input\\') ? 'input' : 'output');
                        videoTuple = [name, dir];
                        console.log("✅ 从字符串解析到:", videoTuple);
                    } catch (e) {
                        console.error("❌ 字符串解析失败:", e);
                    }
                } else {
                    console.error("❌ 无法解析视频数据！data:", data);
                }

                if (videoTuple) {
                    console.log("🎬 准备调用 previewVideo:", videoTuple[0], videoTuple[1]);

                    // 如果有编码警告，先显示警告信息
                    if (codecWarning) {
                        console.warn("⚠️ VideoPreviewNode - 检测到编码警告:", codecWarning);
                        this._codecWarning = codecWarning;
                        this._videoPath = videoPath;
                    }

                    previewVideo(this, videoTuple[0], videoTuple[1]);
                } else {
                    console.error("❌ VideoPreviewNode - 数据格式错误，无法预览！");
                    console.error("❌ 原始数据:", data);
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
