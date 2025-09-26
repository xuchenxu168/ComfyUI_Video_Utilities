import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function fitHeight(node) {
    if (!node || !node.setSize) return;
    
    // 查找视频预览组件
    let videoWidget = node.widgets.find(w => w.name === "videopreview");
    
    if (videoWidget && videoWidget.aspectRatio) {
        // 如果有视频，根据视频尺寸调整节点高度
        let nodeWidth = node.size[0];
        let availableWidth = nodeWidth - 40;
        let videoHeight = availableWidth / videoWidget.aspectRatio;
        
        // 设置更大的高度范围，确保视频完整显示
        let minHeight = 200;
        let maxHeight = Math.min(1200, window.innerHeight * 0.85);
        let finalVideoHeight = Math.max(minHeight, Math.min(maxHeight, videoHeight));
        
        // 计算总节点高度（视频高度 + 其他控件高度 + 边距）
        let totalHeight = finalVideoHeight + 160; // 为其他控件预留更多空间
        
        // 确保最小和最大高度限制
        totalHeight = Math.max(totalHeight, 300);
        totalHeight = Math.min(totalHeight, Math.min(1500, window.innerHeight * 0.9));
        
        // 强制设置节点尺寸
        node.setSize([nodeWidth, totalHeight]);
        
        // 确保画布更新
        if (node.graph) {
            node.graph.setDirtyCanvas(true);
            node.graph.change();
        }
    } else {
        // 如果没有视频，使用默认高度
        let totalHeight = 300;
        for (let widget of node.widgets) {
            if (widget.computeSize) {
                let [width, height] = widget.computeSize(node.size[0]);
                if (height > 0) {
                    totalHeight += height + 15;
                }
            }
        }
        
        totalHeight = Math.max(totalHeight, 300);
        totalHeight = Math.min(totalHeight, 600);
        
        node.setSize([node.size[0], totalHeight]);
    }
    
    // 强制更新画布
    if (node.graph) {
        node.graph.setDirtyCanvas(true);
        node.graph.change();
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
        var el = document.getElementById("loadAFVideo_" + node.id);
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
    element.id = "loadAFVideo_" + node.id;
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
            // 获取节点的实际宽度
            let nodeWidth = node.size[0];
            let availableWidth = nodeWidth - 40; // 考虑左右边距
            
            // 根据视频宽高比计算合适的高度
            let calculatedHeight = availableWidth / this.aspectRatio;
            
            // 设置更大的高度约束，确保视频能够完整显示
            let minHeight = 200;  // 最小高度
            let maxHeight = Math.min(1200, window.innerHeight * 0.85); // 最大高度不超过窗口高度的85%
            
            let finalHeight = Math.max(minHeight, Math.min(maxHeight, calculatedHeight));
            
            // 确保视频预览区域有足够的空间
            this.computedHeight = finalHeight + 60; // 增加更多内边距
            
            return [width, finalHeight];
        }
        return [width, 0]; // 没有视频时返回0高度
    }
    
    previewWidget.value = { hidden: false, paused: false, params: {} }
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "video_preview";
    previewWidget.parentEl.style['width'] = "100%";
    previewWidget.parentEl.setAttribute("data-node-id", node.id);
    element.appendChild(previewWidget.parentEl);
    
    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = true;
    previewWidget.videoEl.loop = false;
    previewWidget.videoEl.muted = false;
    previewWidget.videoEl.style['width'] = "100%";
    previewWidget.videoEl.style['height'] = "auto";
    previewWidget.videoEl.style['display'] = "block";
    previewWidget.videoEl.style['position'] = "relative";
    previewWidget.videoEl.setAttribute("data-node-id", node.id);
    previewWidget.videoEl.setAttribute("preload", "metadata");
    
    // Clear any existing source to prevent ghosting
    previewWidget.videoEl.src = "";
    previewWidget.videoEl.load();
    
    // 增强的事件监听器，确保视频尺寸正确适配
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        if (previewWidget.videoEl.videoWidth && previewWidget.videoEl.videoHeight) {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            
            console.log("Video loaded - Width:", previewWidget.videoEl.videoWidth, "Height:", previewWidget.videoEl.videoHeight, "Aspect Ratio:", previewWidget.aspectRatio);
            
            // 检查是否为竖屏视频（高度大于宽度），如果是则使用更大的高度限制
            const isPortrait = previewWidget.aspectRatio < 1;
            if (isPortrait) {
                console.log("Portrait video detected, using larger height limits");
                // 对于竖屏视频，使用更大的高度限制
                let nodeWidth = previewNode.size[0];
                let availableWidth = nodeWidth - 40;
                let videoHeight = availableWidth / previewWidget.aspectRatio;
                
                // 为竖屏视频使用更大的高度
                let totalHeight = videoHeight + 180;
                totalHeight = Math.max(totalHeight, 400);
                totalHeight = Math.min(totalHeight, Math.min(1800, window.innerHeight * 0.9));
                
                previewNode.setSize([nodeWidth, totalHeight]);
                if (previewNode.graph) {
                    previewNode.graph.setDirtyCanvas(true);
                    previewNode.graph.change();
                }
            } else {
                // 横屏视频使用正常的尺寸调整
                fitHeight(previewNode);
            }
            
            // 延迟多次调整以确保渲染完成
            setTimeout(() => {
                fitHeight(previewNode);
            }, 50);
            
            setTimeout(() => {
                fitHeight(previewNode);
            }, 200);
            
            setTimeout(() => {
                fitHeight(previewNode);
            }, 500);
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

    let params = {
        "filename": file,
        "type": "input",
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

                if (!videoWidget.options.values.includes(path)) {
                    videoWidget.options.values.push(path);
                }

                if (updateNode) {
                    // Use the improved cleanup function
                    clearPreviousVideo(node);
                    
                    videoWidget.value = path;
                    
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

ComfyWidgets.VIDEOPLOAD = videoUpload;

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.LoadAFVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "Load_AF_Video") {
            nodeData.input.required.upload = ["VIDEOPLOAD"];
            
            // 添加窗口大小变化监听器
            if (!window._videoNodeResizeHandler) {
                window._videoNodeResizeHandler = () => {
                    // 重新调整所有视频节点的尺寸
                    if (app.graph && app.graph._nodes_by_id) {
                        const nodes = Object.values(app.graph._nodes_by_id);
                        nodes.forEach(node => {
                            if (node && node.widgets && node.widgets.find(w => w.name === "videopreview")) {
                                try {
                                    fitHeight(node);
                                } catch (error) {
                                    console.log("Error resizing video node:", error);
                                }
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
                
                // 添加手动调整尺寸选项
                menuOptions.push({
                    content: "调整视频尺寸",
                    callback: () => {
                        if (this._forceResize) {
                            this._forceResize();
                        } else {
                            fitHeight(this);
                        }
                    }
                });
                
                // 添加强制大尺寸选项
                menuOptions.push({
                    content: "强制大尺寸显示",
                    callback: () => {
                        const videoWidget = this.widgets.find(w => w.name === "videopreview");
                        if (videoWidget && videoWidget.aspectRatio) {
                            let nodeWidth = this.size[0];
                            let availableWidth = nodeWidth - 40;
                            let videoHeight = availableWidth / videoWidget.aspectRatio;
                            
                            // 使用更大的高度限制
                            let totalHeight = videoHeight + 200;
                            totalHeight = Math.max(totalHeight, 400);
                            totalHeight = Math.min(totalHeight, Math.min(2000, window.innerHeight * 0.95));
                            
                            this.setSize([nodeWidth, totalHeight]);
                            if (this.graph) {
                                this.graph.setDirtyCanvas(true);
                                this.graph.change();
                            }
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