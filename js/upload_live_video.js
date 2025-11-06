import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

// VERSION: 2025-01-08-WORKING
console.log("🎬 Upload Live Video JS loaded - WORKING VERSION");

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
    }
    
    .video_preview video {
        width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        background: #1a1a1a;
        margin: 0;
        padding: 0;
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
    if (!node || !node.setSize) return;
    
    // 查找视频预览组件
    let videoWidget = node.widgets.find(w => w.name === "videopreview");
    
    if (videoWidget && videoWidget.aspectRatio) {
        // 简单直接：根据视频尺寸调整节点
        let nodeWidth = node.size[0];
        let videoHeight = (nodeWidth - 40) / videoWidget.aspectRatio;
        
        // 根据视频高度动态调整空白空间
        let extraSpace = 0;
        if (videoHeight > 400) {
            extraSpace = 50; // 高视频增加50px空白
        } else if (videoHeight > 300) {
            extraSpace = 30; // 中等高度视频增加30px空白
        } else if (videoHeight > 200) {
            extraSpace = 20; // 较低视频增加20px空白
        }
        
        let totalHeight = videoHeight + 200 + extraSpace; // 视频高度 + 控件空间 + 动态空白
        
        // 设置节点尺寸
        node.setSize([nodeWidth, totalHeight]);
        
        // 更新画布
        if (node.graph) {
            node.graph.setDirtyCanvas(true);
            node.graph.change();
        }
    }
}

function clearPreviousVideo(node) {
    // Remove only video preview widgets (name === 'videopreview')
    // Keep the original widgets (video dropdown and upload button)
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const widget = node.widgets[i];
        if (widget.name === 'videopreview') {
            if (widget.parentEl) {
                try {
                    widget.parentEl.remove();
                } catch (error) {
                    console.log("Error removing widget parent:", error);
                }
            }
            node.widgets.splice(i, 1);
        }
    }
    
    // Remove any existing video elements by ID
    try {
        var el = document.getElementById("uploadliveVideo_" + node.id);
        if (el) {
            el.remove();
        }
    } catch (error) {
        console.log("Error removing video element:", error);
    }
    
    // Clear any remaining video elements that might be orphaned
    const existingVideos = document.querySelectorAll(`[data-node-id="${node.id}"]`);
    existingVideos.forEach(video => {
        try {
            video.remove();
        } catch (error) {
            console.log("Error removing orphaned video:", error);
        }
    });
}

function previewVideo(node, file) {
    // 防止重复调用 - 如果正在加载相同的视频，直接返回
    if (node._currentVideoFile === file && node._videoLoading) {
        return;
    }

    // 取消之前的异步操作
    if (node._abortController) {
        node._abortController.abort();
    }
    node._abortController = new AbortController();

    // 标记正在加载
    node._currentVideoFile = file;
    node._videoLoading = true;

    // Clear previous video content completely
    clearPreviousVideo(node);
    
    var element = document.createElement("div");
    element.id = "uploadliveVideo_" + node.id;
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
    
    previewWidget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            // 简单直接：根据视频宽高比计算高度
            let videoHeight = (width - 40) / this.aspectRatio;
            return [width, videoHeight];
        }
        return [width, 0];
    }
    
    previewWidget.value = { hidden: false, paused: false, params: {} }
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "video_preview";
    previewWidget.parentEl.style['width'] = "100%";
    previewWidget.parentEl.style['margin'] = "0";
    previewWidget.parentEl.style['padding'] = "0";
    previewWidget.parentEl.setAttribute("data-node-id", node.id);
    element.appendChild(previewWidget.parentEl);
    
    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = true;
    previewWidget.videoEl.loop = false;
    previewWidget.videoEl.muted = false;
    previewWidget.videoEl.autoplay = false; // 禁用自动播放，由用户控制
    previewWidget.videoEl.style['width'] = "100%";
    previewWidget.videoEl.style['minHeight'] = "200px";
    previewWidget.videoEl.style['height'] = "auto";
    previewWidget.videoEl.style['display'] = "block";
    previewWidget.videoEl.style['position'] = "relative";
    previewWidget.videoEl.style['backgroundColor'] = "#000";
    previewWidget.videoEl.style['margin'] = "0";
    previewWidget.videoEl.style['padding'] = "0";
    previewWidget.videoEl.setAttribute("data-node-id", node.id);
    previewWidget.videoEl.setAttribute("preload", "metadata");
    
    // Clear any existing source to prevent ghosting
    previewWidget.videoEl.src = "";
    previewWidget.videoEl.load();
    
    // 简单的事件监听器
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        // 标记加载完成
        node._videoLoading = false;

        if (previewWidget.videoEl.videoWidth && previewWidget.videoEl.videoHeight) {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(previewNode);
        }
    });
    
    // 添加canplay事件监听器，确保视频可以播放时再次调整尺寸
    previewWidget.videoEl.addEventListener("canplay", () => {
        if (previewWidget.aspectRatio) {
            console.log("Video can play - adjusting size");
            fitHeight(previewNode);
        }
    });
    
    // 添加loadeddata事件监听器
    previewWidget.videoEl.addEventListener("loadeddata", () => {
        if (previewWidget.aspectRatio) {
            console.log("Video data loaded - adjusting size");
            fitHeight(previewNode);
        }
    });
    
    previewWidget.videoEl.addEventListener("error", (e) => {
        // 标记加载完成（即使失败）
        node._videoLoading = false;

        console.error("❌ Upload_Live_Video: Video loading error:", e);
        console.error("❌ Upload_Live_Video: Video error details:", previewWidget.videoEl.error);
        console.error("❌ Upload_Live_Video: Video src:", previewWidget.videoEl.src);
        console.error("❌ Upload_Live_Video: File:", file);

        // 不再显示 Topaz 特定的错误提示，因为我们已经有转码功能了
        // 只是隐藏视频元素
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
    });

    let actualFilename = file;
    let fileType = "input";

    // 处理前缀格式：[Output] filename 或 [Input] filename
    if (file.startsWith("[Output] ")) {
        actualFilename = file.substring(9);
        fileType = "output";
    } else if (file.startsWith("[Input] ")) {
        actualFilename = file.substring(8);
        fileType = "input";
    }
    // 处理后缀格式：filename [output] 或 filename [input]（upload widget 格式）
    else if (file.endsWith(" [output]")) {
        actualFilename = file.substring(0, file.length - 9);
        fileType = "output";
    } else if (file.endsWith(" [input]")) {
        actualFilename = file.substring(0, file.length - 8);
        fileType = "input";
    } else if (file.startsWith("--- ") || file === "No video files found") {
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
        return;
    }

    // 处理子文件夹路径（例如 "sora_videos/video.mp4"）
    let subfolder = "";
    let filename = actualFilename;
    if (actualFilename.includes("/")) {
        const parts = actualFilename.split("/");
        filename = parts.pop(); // 最后一部分是文件名
        subfolder = parts.join("/"); // 其余部分是子文件夹
    }

    console.log("🎬 Upload_Live_Video 路径解析:");
    console.log("   - file:", file);
    console.log("   - actualFilename:", actualFilename);
    console.log("   - filename:", filename);
    console.log("   - subfolder:", subfolder);
    console.log("   - fileType:", fileType);

    let params = {
        "filename": filename,
        "type": fileType,
    }

    // 只有当 subfolder 不为空时才添加
    if (subfolder) {
        params.subfolder = subfolder;
    }

    console.log("   - params:", params);

    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    previewWidget.videoEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;

    let target_width = 256;
    if (element.style?.width) {
        target_width = element.style.width.slice(0, -2) * 2;
    }

    if (!params.force_size || params.force_size.includes("?") || params.force_size == "Disabled") {
        params.force_size = target_width + "x?";
    } else {
        let size = params.force_size.split("x");
        let ar = parseInt(size[0]) / parseInt(size[1]);
        params.force_size = target_width + "x" + (target_width / ar);
    }

    // Set video source and append to parent
    // 智能选择端点：通过 API 检测视频编码，MPEG-4 视频使用转码
    params._t = Date.now();

    // 异步检测编码并设置视频源
    (async () => {
        try {
            // 检查是否已被取消
            if (node._abortController.signal.aborted) {
                console.log("🎬 Upload_Live_Video: Async operation aborted");
                return;
            }

            // 调用编码检测 API
            const detectParams = {
                filename: filename,
                type: params.type || 'input'
            };
            // 如果有 subfolder，也传递给 API
            if (subfolder) {
                detectParams.subfolder = subfolder;
            }
            const detectUrl = api.apiURL('/video_utilities/detect_codec?' + new URLSearchParams(detectParams));

            console.log("🎬 Upload_Live_Video: Detecting codec...");
            console.log("   - detectUrl:", detectUrl);
            console.log("   - detectParams:", detectParams);

            const response = await fetch(detectUrl, { signal: node._abortController.signal });
            const data = await response.json();

            const needsTranscode = data.needs_transcode || false;
            const codec = data.codec || 'unknown';

            // 再次检查是否已被取消
            if (node._abortController.signal.aborted) {
                console.log("🎬 Upload_Live_Video: Async operation aborted before setting src");
                return;
            }

            const endpoint = needsTranscode ? '/video_utilities/viewvideo' : '/view';
            const videoUrl = api.apiURL(endpoint + '?' + new URLSearchParams(params));

            console.log("🎬 Upload_Live_Video: File:", filename);
            console.log("🎬 Upload_Live_Video: Codec:", codec);
            console.log("🎬 Upload_Live_Video: Needs transcode:", needsTranscode);
            console.log("🎬 Upload_Live_Video: Using endpoint:", endpoint);
            console.log("🎬 Upload_Live_Video: Video URL:", videoUrl);

            // 先设置 src
            previewWidget.videoEl.src = videoUrl;
            // 强制禁用自动播放
            previewWidget.videoEl.autoplay = false;
            // 然后添加到 DOM（模仿备份文件的做法）
            previewWidget.videoEl.hidden = false;
            previewWidget.parentEl.appendChild(previewWidget.videoEl);
        } catch (error) {
            // 忽略 AbortError（操作被取消）
            if (error.name === 'AbortError') {
                console.log("🎬 Upload_Live_Video: Fetch aborted");
                return;
            }

            console.warn("⚠️ Upload_Live_Video: Codec detection failed, using /video_utilities/viewvideo for safety:", error);

            // 检查是否已被取消
            if (node._abortController.signal.aborted) {
                return;
            }

            // 如果检测失败，使用转码端点以确保兼容性（特别是对于 Topaz 视频）
            const videoUrl = api.apiURL('/video_utilities/viewvideo?' + new URLSearchParams(params));
            console.log("🎬 Upload_Live_Video: Fallback URL:", videoUrl);
            // 先设置 src
            previewWidget.videoEl.src = videoUrl;
            // 强制禁用自动播放
            previewWidget.videoEl.autoplay = false;
            // 然后添加到 DOM
            previewWidget.videoEl.hidden = false;
            previewWidget.parentEl.appendChild(previewWidget.videoEl);
        }
    })();

    // 添加错误处理
    previewWidget.videoEl.onerror = function(e) {
        console.error("🎬 Upload_Live_Video: Video load error:", e);
        console.error("🎬 Upload_Live_Video: Video src:", previewWidget.videoEl.src);
        console.error("🎬 Upload_Live_Video: Video error code:", previewWidget.videoEl.error?.code);
        console.error("🎬 Upload_Live_Video: Video error message:", previewWidget.videoEl.error?.message);
    };

    previewWidget.videoEl.onloadedmetadata = function() {
        console.log("🎬 Upload_Live_Video: Video metadata loaded successfully");
        console.log("🎬 Upload_Live_Video: Video duration:", previewWidget.videoEl.duration);
        console.log("🎬 Upload_Live_Video: Video dimensions:", previewWidget.videoEl.videoWidth, "x", previewWidget.videoEl.videoHeight);
    };
    
    // 强制多次更新尺寸以确保正确渲染
    setTimeout(() => {
        fitHeight(previewNode);
    }, 50);
    
    setTimeout(() => {
        fitHeight(previewNode);
    }, 150);
    
    setTimeout(() => {
        fitHeight(previewNode);
    }, 300);
    
    // 添加一个强制刷新函数
    const forceResize = () => {
        if (previewWidget.aspectRatio) {
            console.log("Force resizing node");
            fitHeight(previewNode);
        }
    };
    
    // 存储强制刷新函数供后续使用
    if (!node._forceResize) {
        node._forceResize = forceResize;
    }
    
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

function videoUpload(node, inputName, inputData, app) {
    const videoWidget = node.widgets.find((w) => w.name === "video");
    let uploadWidget;
    
    var default_value = videoWidget.value;
    Object.defineProperty(videoWidget, "value", {
        set : function(value) {
            this._real_value = value;
        },

        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });
    
    async function uploadFile(file, updateNode, pasted = false) {
        try {
            const body = new FormData();
            body.append("image", file);
            if (pasted) body.append("subfolder", "pasted");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                let path = data.name;
                if (data.subfolder) path = data.subfolder + "/" + path;

                let formattedPath = `[Input] ${path}`;
                
                if (!videoWidget.options.values.includes(formattedPath)) {
                    videoWidget.options.values.push(formattedPath);
                }

                if (updateNode) {
                    // Use the improved cleanup function
                    clearPreviousVideo(node);
                    
                    videoWidget.value = formattedPath;
                    
                    if (videoWidget.callback) {
                        videoWidget.callback();
                    }
                }
            } else {
                alert(resp.status + " - " + resp.statusText);
            }
        } catch (error) {
            alert(error);
        }
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: "video/webm,video/mp4,video/mkv,video/avi",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], true);
            }
        },
    });
    document.body.append(fileInput);

    uploadWidget = node.addWidget("button", "choose video file to upload", "Video", () => {
        fileInput.click();
    });

    uploadWidget.serialize = false;

    // 不在这里设置 callback，让 onNodeCreated 统一处理
    // 这样避免重复调用 previewVideo

    return { widget: uploadWidget };
}

ComfyWidgets.VIDEOPLOAD_LIVE = videoUpload;

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.UploadLiveVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "VideoUtilitiesUploadLiveVideo" || nodeData?.name == "Upload_Live_Video") {
            nodeData.input.required.upload = ["VIDEOPLOAD_LIVE"];

            // 拦截节点创建，添加视频预览
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 找到 video 下拉列表 widget
                const videoWidget = this.widgets.find(w => w.name === "video");
                if (videoWidget) {
                    const node = this; // 保存节点引用

                    // 使用 chainCallback 方式，不覆盖原有 callback
                    const originalCallback = videoWidget.callback;
                    videoWidget.callback = function() {
                        // 先调用原始 callback
                        let r;
                        if (originalCallback) {
                            r = originalCallback.apply(this, arguments);
                        }

                        // 然后执行我们的预览逻辑
                        clearPreviousVideo(node);
                        if (videoWidget.value &&
                            !videoWidget.value.startsWith("--- ") &&
                            videoWidget.value !== "No video files found") {
                            previewVideo(node, videoWidget.value);
                        }

                        return r;
                    };

                    // 初始化时也创建预览（跳过无效值）
                    // 延迟执行，确保 widget 完全初始化
                    setTimeout(() => {
                        if (videoWidget.value &&
                            !videoWidget.value.startsWith("--- ") &&
                            videoWidget.value !== "No video files found") {
                            previewVideo(node, videoWidget.value);
                        }
                    }, 100);
                }

                return result;
            };

            // 添加窗口大小变化监听器
            if (!window._videoNodeResizeHandler) {
                window._videoNodeResizeHandler = () => {
                    // 重新调整所有视频节点的尺寸
                    if (app.graph && app.graph._nodes_by_id) {
                        const nodes = Object.values(app.graph._nodes_by_id);
                        nodes.forEach(node => {
                            if (node.widgets && node.widgets.find(w => w.name === "videopreview")) {
                                fitHeight(node);
                            }
                        });
                    }
                };
                window.addEventListener('resize', window._videoNodeResizeHandler);
            }
            
            // 添加右键菜单选项
            const originalGetExtraMenuOptions = nodeData.prototype.getExtraMenuOptions;
            nodeData.prototype.getExtraMenuOptions = function(_, options) {
                const menuOptions = [];
                
                // 添加调整尺寸选项
                menuOptions.push({
                    content: "调整视频尺寸",
                    callback: () => {
                        fitHeight(this);
                    }
                });
                
                // 添加强制大尺寸选项（为Upload Live Video节点特别优化）
                menuOptions.push({
                    content: "强制大尺寸显示",
                    callback: () => {
                        const videoWidget = this.widgets.find(w => w.name === "videopreview");
                        if (videoWidget && videoWidget.aspectRatio) {
                            let nodeWidth = this.size[0];
                            let availableWidth = nodeWidth - 40;
                            let videoHeight = availableWidth / videoWidget.aspectRatio;
                            
                            // 为Upload Live Video节点使用更大的高度限制（为5个输出端口预留空间）
                            let totalHeight = videoHeight + 250;
                            totalHeight = Math.max(totalHeight, 400); // 最小高度减少100
                            totalHeight = Math.min(totalHeight, Math.min(3150, window.innerHeight * 0.98)); // 最大高度再增加100
                            
                            this.setSize([nodeWidth, totalHeight]);
                            if (this.graph) {
                                this.graph.setDirtyCanvas(true);
                                this.graph.change();
                            }
                        }
                    }
                });
                
                // 添加超大型尺寸选项（专门为Upload Live Video节点）
                menuOptions.push({
                    content: "超大型尺寸显示",
                    callback: () => {
                        const videoWidget = this.widgets.find(w => w.name === "videopreview");
                        if (videoWidget && videoWidget.aspectRatio) {
                            let nodeWidth = this.size[0];
                            let availableWidth = nodeWidth - 40;
                            let videoHeight = availableWidth / videoWidget.aspectRatio;
                            
                            // 使用超大的高度限制，确保任何视频都能完整显示
                            let totalHeight = videoHeight + 300;
                            totalHeight = Math.max(totalHeight, 500);
                            totalHeight = Math.min(totalHeight, Math.min(5000, window.innerHeight * 0.99));
                            
                            this.setSize([nodeWidth, totalHeight]);
                            if (this.graph) {
                                this.graph.setDirtyCanvas(true);
                                this.graph.change();
                            }
                        }
                    }
                });
                
                // 添加调试选项
                menuOptions.push({
                    content: "调试视频尺寸",
                    callback: () => {
                        const videoWidget = this.widgets.find(w => w.name === "videopreview");
                        if (videoWidget && videoWidget.aspectRatio) {
                            console.log("=== 视频尺寸调试信息 ===");
                            console.log("节点宽度:", this.size[0]);
                            console.log("视频宽高比:", videoWidget.aspectRatio);
                            console.log("计算出的视频高度:", (this.size[0] - 40) / videoWidget.aspectRatio);
                            console.log("当前节点高度:", this.size[1]);
                            console.log("视频元素实际尺寸:", videoWidget.videoEl?.videoWidth, "x", videoWidget.videoEl?.videoHeight);
                        }
                    }
                });
                
                // 添加刷新选项
                menuOptions.push({
                    content: "刷新视频预览",
                    callback: () => {
                        const videoWidget = this.widgets.find(w => w.name === "video");
                        if (videoWidget && videoWidget.callback) {
                            videoWidget.callback();
                        }
                    }
                });
                
                // 调用原始菜单选项
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.call(this, _, options);
                }
                
                // 将新选项添加到菜单开头
                options.unshift(...menuOptions);
            };
            
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
    },
});
