// Color Picker Widget for ComfyUI
// 为 Color_Picker 节点添加可视化颜色选择器

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.ColorPicker",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== "Color_Picker") return;
        
        console.log("[Video_Utilities] Registering Color Picker widget");
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            
            // 找到 color 输入框
            const colorWidget = this.widgets?.find(w => w.name === "color");
            if (!colorWidget) {
                console.warn("[Video_Utilities] Color widget not found");
                return result;
            }
            
            // 保存原始值
            const originalValue = colorWidget.value || "#FFFF00";
            
            // 创建颜色选择器容器
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative;
                display: inline-block;
                margin: 5px 0;
            `;
            
            // 创建颜色预览框
            const colorPreview = document.createElement("div");
            colorPreview.style.cssText = `
                width: 40px;
                height: 40px;
                border: 2px solid #666;
                border-radius: 4px;
                cursor: pointer;
                background-color: ${originalValue};
                display: inline-block;
                vertical-align: middle;
                margin-right: 10px;
            `;
            
            // 创建隐藏的颜色输入框（HTML5 color picker）
            const colorInput = document.createElement("input");
            colorInput.type = "color";
            colorInput.value = originalValue;
            colorInput.style.cssText = `
                position: absolute;
                opacity: 0;
                width: 0;
                height: 0;
            `;
            
            // 创建文本输入框
            const textInput = document.createElement("input");
            textInput.type = "text";
            textInput.value = originalValue;
            textInput.style.cssText = `
                width: 100px;
                padding: 8px;
                border: 1px solid #666;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: #fff;
                font-family: monospace;
                font-size: 14px;
                vertical-align: middle;
            `;
            
            // 更新颜色的函数
            const updateColor = (newColor) => {
                // 确保颜色格式正确
                if (!newColor.startsWith('#')) {
                    newColor = '#' + newColor;
                }
                newColor = newColor.toUpperCase();
                
                // 更新所有显示
                colorPreview.style.backgroundColor = newColor;
                colorInput.value = newColor;
                textInput.value = newColor;
                colorWidget.value = newColor;
                
                // 触发节点更新
                if (this.onResize) {
                    this.onResize(this.size);
                }
            };
            
            // 颜色选择器变化事件
            colorInput.addEventListener("input", (e) => {
                updateColor(e.target.value);
            });
            
            // 文本输入框变化事件
            textInput.addEventListener("input", (e) => {
                let value = e.target.value.trim();
                // 验证颜色格式
                if (/^#?[0-9A-Fa-f]{6}$/.test(value)) {
                    updateColor(value);
                }
            });
            
            // 文本输入框失去焦点时验证
            textInput.addEventListener("blur", (e) => {
                let value = e.target.value.trim();
                if (!/^#?[0-9A-Fa-f]{6}$/.test(value)) {
                    // 如果格式不正确，恢复到当前值
                    textInput.value = colorWidget.value;
                }
            });
            
            // 点击预览框打开颜色选择器
            colorPreview.addEventListener("click", () => {
                colorInput.click();
            });
            
            // 组装容器
            container.appendChild(colorInput);
            container.appendChild(colorPreview);
            container.appendChild(textInput);
            
            // 隐藏原始输入框
            colorWidget.type = "hidden";
            
            // 添加自定义 widget
            const pickerWidget = this.addDOMWidget("colorpicker", "picker", container);
            pickerWidget.serialize = false; // 不序列化这个 widget
            
            // 同步原始 widget 的值到颜色选择器
            const originalCallback = colorWidget.callback;
            colorWidget.callback = function(value) {
                updateColor(value);
                if (originalCallback) {
                    originalCallback.call(this, value);
                }
            };
            
            console.log("[Video_Utilities] Color Picker widget created");
            
            return result;
        };
    },
    
    async loadedGraphNode(node, app) {
        if (node.type !== "Color_Picker") return;
        
        // 当从保存的工作流加载时，确保颜色选择器显示正确的值
        const colorWidget = node.widgets?.find(w => w.name === "color");
        if (colorWidget && colorWidget.value) {
            // 触发更新
            setTimeout(() => {
                const pickerWidget = node.widgets?.find(w => w.name === "colorpicker");
                if (pickerWidget && pickerWidget.element) {
                    const textInput = pickerWidget.element.querySelector('input[type="text"]');
                    const colorInput = pickerWidget.element.querySelector('input[type="color"]');
                    const colorPreview = pickerWidget.element.querySelector('div');
                    
                    if (textInput && colorInput && colorPreview) {
                        const value = colorWidget.value;
                        textInput.value = value;
                        colorInput.value = value;
                        colorPreview.style.backgroundColor = value;
                    }
                }
            }, 100);
        }
    }
});

