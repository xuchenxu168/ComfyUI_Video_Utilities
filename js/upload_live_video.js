import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

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
    // Remove all existing video preview widgets
    while (node.widgets.length > 2) {
        let widget = node.widgets.pop();
        if (widget.parentEl) {
            try {
                widget.parentEl.remove();
            } catch (error) {
                console.log("Error removing widget parent:", error);
            }
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
    previewWidget.videoEl.style['width'] = "100%";
    previewWidget.videoEl.style['height'] = "auto"; // 让视频保持原始比例
    previewWidget.videoEl.style['display'] = "block";
    previewWidget.videoEl.style['position'] = "relative";
    previewWidget.videoEl.style['object-fit'] = "contain"; // 保持比例
    previewWidget.videoEl.style['margin'] = "0";
    previewWidget.videoEl.style['padding'] = "0";
    previewWidget.videoEl.setAttribute("data-node-id", node.id);
    previewWidget.videoEl.setAttribute("preload", "metadata");
    
    // Clear any existing source to prevent ghosting
    previewWidget.videoEl.src = "";
    previewWidget.videoEl.load();
    
    // 简单的事件监听器
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
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
        console.log("Video loading error:", e);
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
    });

    let actualFilename = file;
    let fileType = "input";
    if (file.startsWith("[Output] ")) {
        actualFilename = file.substring(9);
        fileType = "output";
    } else if (file.startsWith("[Input] ")) {
        actualFilename = file.substring(8);
        fileType = "input";
    } else if (file.startsWith("--- ") || file === "No video files found") {
        previewWidget.parentEl.hidden = true;
        fitHeight(previewNode);
        return;
    }
    
    let params = {
        "filename": actualFilename,
        "type": fileType,
    }
    
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
    previewWidget.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
    previewWidget.videoEl.hidden = false;
    previewWidget.parentEl.appendChild(previewWidget.videoEl);
    
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

    previewVideo(node, videoWidget.value);
    const cb = node.callback;
    videoWidget.callback = function () {
        previewVideo(node, videoWidget.value);
        if (cb) {
            return cb.apply(this, arguments);
        }
    };

    return { widget: uploadWidget };
}

ComfyWidgets.VIDEOPLOAD_LIVE = videoUpload;

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.UploadLiveVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "Upload_Live_Video") {
            nodeData.input.required.upload = ["VIDEOPLOAD_LIVE"];
            
            // 添加窗口大小变化监听器
            if (!window._videoNodeResizeHandler) {
                window._videoNodeResizeHandler = () => {
                    // 重新调整所有视频节点的尺寸
                    app.graph._nodes_by_id.forEach(node => {
                        if (node.widgets && node.widgets.find(w => w.name === "videopreview")) {
                            fitHeight(node);
                        }
                    });
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
