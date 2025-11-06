# ComfyUI Video Utilities

<div align="center">

**功能强大的 ComfyUI 视频处理与字幕生成扩展**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 📋 目录

- [功能特性](#功能特性)
- [安装指南](#安装指南)
- [节点列表](#节点列表)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [详细文档](#详细文档)

---

### 🎯 功能特性

#### 🎤 AI 语音识别与字幕生成
- **多引擎支持**：faster-whisper、Transformers
- **多语言识别**：中文、英文、日语等
- **智能字幕**：静态字幕、动态字幕（逐词显示）、滚动字幕
- **专业效果**：11 种动画效果、描边、竖排文字、自适应字体
- **高级功能**：参考文本校正、提示词引导、逐词时间戳

#### 🎬 视频处理
- **视频拼接**：支持 8 个视频拼接，多种拼接模式（concat、hstack、vstack、grid）
- **视频转 GIF**：高质量转换，支持质量控制和大小优化
- **帧提取**：提取首帧/尾帧，支持偏移量
- **格式转换**：自动检测编码，智能转码（MPEG-4 → H.264）

#### 📺 视频预览与加载
- **实时预览**：浏览器内视频预览，支持 Topaz 等特殊编码
- **文件上传**：拖拽上传视频文件
- **目录监控**：自动监控 output 目录最新视频
- **双向加载**：支持 input 和 output 目录

#### 🛠️ 工具节点
- **颜色选择器**：可视化颜色选择
- **文本节点**：通用字符串输出
- **纯色图片**：生成任意颜色的图片

---

### 📦 安装指南

#### 方法 1：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager
2. 搜索 "Video Utilities"
3. 点击安装

#### 方法 2：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI_Video_Utilities.git
cd ComfyUI_Video_Utilities
pip install -r requirements.txt
```

#### 依赖说明

**核心依赖**（必需）：
```bash
pip install -r requirements.txt
```

**可选依赖**：
- 字幕功能：`pip install -r requirements_subtitle.txt`
- 人脸修复：`pip install -r requirements_face_restore.txt`

#### 系统要求

- **Python**: 3.8+
- **FFmpeg**: 必需（用于视频处理）
- **CUDA**: 可选（用于 GPU 加速 ASR）

---

### 📚 节点列表

#### 🎤 AI 语音识别与字幕（新节点）

| 节点名称 | 功能描述 | 主要用途 |
|---------|---------|---------|
| **Audio_To_Text** 🎤 | 将视频/音频转录为文本 | 语音识别、字幕生成 |
| **Text_To_Video_Static** 📝 | 添加静态字幕（整句显示） | 电影字幕、教学视频 |
| **Text_To_Video_Dynamic** 🎯 | 添加动态字幕（逐词显示） | 抖音风格字幕、卡拉OK |
| **Text_To_Video_Scrolling** 🎬 | 添加滚动字幕 | 电影片尾、新闻滚动 |
| **Color_Picker** 🎨 | 颜色选择器 | 字幕颜色选择 |

#### 🎬 视频处理节点

| 节点名称 | 功能描述 | 主要用途 |
|---------|---------|---------|
| **Video_Stitching** | 拼接多个视频（最多 8 个） | 视频合并、网格展示 |
| **Video_To_GIF** | 视频转 GIF 动图 | 表情包制作、社交分享 |
| **Get_Last_Frame** | 提取视频最后一帧 | 缩略图生成、帧分析 |
| **Get_First_Frame** | 提取视频第一帧 | 封面生成、帧分析 |

#### 📺 视频预览与加载节点

| 节点名称 | 功能描述 | 主要用途 |
|---------|---------|---------|
| **Video_Preview** | 浏览器内视频预览 | 视频预览、编码检测 |
| **Upload_Live_Video** | 上传本地视频文件 | 导入视频、快速测试 |
| **Load_AF_Video** | 从目录加载视频 | 加载已有视频、批量处理 |
| **Live_Video_Monitor** | 监控最新视频 | 实时监控、自动化流程 |
| **Preview_GIF** | 预览 GIF 动图 | GIF 预览、质量检查 |

#### 🛠️ 工具节点

| 节点名称 | 功能描述 | 主要用途 |
|---------|---------|---------|
| **Prompt_Text_Node** | 文本输出节点 | 提示词输入、文本传递 |
| **RGB_Empty_Image** | 生成纯色图片 | 背景图生成、颜色测试 |
| **Get VHS File Path** | 获取 VHS 视频路径 | VHS 视频处理 |
| **Audio_To_Subtitle**（旧版） | 音频转字幕（一体化） | 快速字幕生成 |

---

### 💡 使用示例

#### 示例 1：基础字幕生成工作流

```
Upload_Live_Video → Audio_To_Text → Text_To_Video_Static → Video_Preview
```

**步骤**：
1. 使用 `Upload_Live_Video` 上传视频
2. 使用 `Audio_To_Text` 识别语音（选择 Belle-whisper-large-v3-zh-punct-ct2 模型）
3. 使用 `Text_To_Video_Static` 添加静态字幕（选择字体、颜色、动画）
4. 使用 `Video_Preview` 预览结果

---

#### 示例 2：动态字幕（抖音风格）

```
Load_AF_Video → Audio_To_Text → Text_To_Video_Dynamic → Video_Preview
```

**步骤**：
1. 使用 `Load_AF_Video` 加载视频
2. 使用 `Audio_To_Text` 识别语音（输出逐词时间戳）
3. 使用 `Text_To_Video_Dynamic` 添加动态字幕
   - 设置 `max_lines=3`（最多显示 3 行）
   - 设置 `clearance_threshold=2.0`（静音 2 秒清空字幕）
4. 使用 `Video_Preview` 预览结果

---

#### 示例 3：电影片尾滚动字幕

```
Load_AF_Video → Text_To_Video_Scrolling → Video_Preview
```

**步骤**：
1. 使用 `Load_AF_Video` 加载视频
2. 使用 `Text_To_Video_Scrolling` 添加滚动字幕
   - 设置 `scroll_type=vertical_up`（垂直向上滚动）
   - 设置 `scroll_speed=50`（滚动速度）
3. 使用 `Video_Preview` 预览结果

---

#### 示例 4：视频拼接

```
Load_AF_Video (x3) → Video_Stitching → Video_Preview
```

**步骤**：
1. 使用 3 个 `Load_AF_Video` 节点加载 3 个视频
2. 使用 `Video_Stitching` 拼接视频
   - 设置 `stitch_method=concat`（顺序拼接）
   - 设置 `transition_type=fade`（淡入淡出转场）
3. 使用 `Video_Preview` 预览结果

---

### ❓ 常见问题

#### Q1: 视频预览显示黑屏怎么办？

**A**: 这通常是视频编码问题。解决方法：
1. 检查浏览器控制台是否有错误信息
2. 确认 FFmpeg 已正确安装
3. 使用 `Video_Preview` 节点会自动检测编码并转码
4. 如果是 Topaz 等特殊编码，系统会自动转码为 H.264

#### Q2: ASR 识别不准确怎么办？

**A**: 提高识别准确度的方法：
1. 使用 `Belle-whisper-large-v3-zh-punct-ct2` 模型（中文优化）
2. 使用 `prompt` 参数提供专业术语或上下文
3. 使用 `reference_text` 参数提供参考文本进行自动校正
4. 确保音频质量清晰，无噪音

#### Q3: 字幕位置不对怎么调整？

**A**: 调整字幕位置的方法：
1. 使用 `position` 参数选择预设位置（9 个位置可选）
2. 使用 `offset_x` 和 `offset_y` 参数微调位置
3. 例如：`offset_y=-50` 可以将字幕向上移动 50 像素

#### Q4: 字体文件放在哪里？

**A**: 字体文件位置：
1. 将字体文件（.ttf、.ttc、.otf）放入 `ComfyUI_Video_Utilities/Fonts/` 目录
2. 重启 ComfyUI 后会自动识别
3. 推荐使用中文字体：思源黑体、微软雅黑等

#### Q5: FFmpeg 如何安装？

**A**: FFmpeg 安装方法：
- **Windows**: 下载 FFmpeg 并添加到系统 PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

### 📖 详细文档

更多详细信息请参考以下文档：

- [安装指南](INSTALLATION_GUIDE.md) - 详细的安装步骤和依赖说明
- [快速开始](QUICKSTART_WORKFLOWS.md) - 快速上手工作流示例
- [动画效果指南](ANIMATION_GUIDE.md) - 11 种字幕动画效果详解
- [滚动字幕指南](SCROLLING_SUBTITLE_GUIDE.md) - 滚动字幕使用指南
- [工作流示例](WORKFLOW_EXAMPLES.md) - 完整工作流示例
- [节点设计文档](NODE_DESIGN.md) - 节点设计和参数说明

---

### 📝 更新日志

#### v2.0.0 (2024-11)
- ✨ 新增 `Audio_To_Text` 节点（多引擎 ASR）
- ✨ 新增 `Text_To_Video_Static` 节点（静态字幕）
- ✨ 新增 `Text_To_Video_Dynamic` 节点（动态字幕）
- ✨ 新增 `Text_To_Video_Scrolling` 节点（滚动字幕）
- ✨ 新增 `Color_Picker` 节点（颜色选择器）
- 🔧 修复视频预览黑屏问题
- 🔧 优化编码检测和自动转码
- 🔧 改进字幕渲染性能

#### v1.0.0 (2024-10)
- 🎉 初始版本发布
- ✨ 视频拼接功能
- ✨ 视频转 GIF 功能
- ✨ 视频预览功能
- ✨ 帧提取功能

---

### 📄 许可证

MIT License

---

### 🙏 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的 AI 图像生成框架
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效的 Whisper 实现
- [FFmpeg](https://ffmpeg.org/) - 视频处理工具
- [Belle-whisper](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct) - 中文优化的 Whisper 模型

---

<div align="center">

**Made with ❤️ by Ken-Chen**

⭐ If you find this project helpful, please give it a star! ⭐

</div>

