# ComfyUI Video Utilities

一个功能强大的 ComfyUI 自定义节点扩展，专注于视频处理和 GIF 转换功能。

## 📋 功能特性

### 🎬 视频处理节点
- **Get VHS File Path** - 从 Video Combine 节点输出中提取视频文件路径
- **Video to GIF** - 高质量视频转 GIF 转换，支持多种质量控制选项
- **Preview GIF** - 在浏览器或媒体播放器中预览 GIF 文件
- **Video Preview** - 在 ComfyUI 界面中预览视频文件
- **Upload Live Video** - 上传和加载视频文件，含前端预览与上传控件；输出 `VHS_FILENAMES`
- **Load AF Video** - 从 input 目录加载音视频，含前端预览与上传控件
- **Live Video Monitor** - 监控 output 目录最新视频，自动刷新并预览；输出 `VHS_FILENAMES`
- **Video Stitching** - 视频拼接功能
- **Get Last Frame** - 提取视频最后一帧及尾帧往前的任意帧
- **Prompt Text Node** - 通用字符串输出（STRING）
- **RGB Empty Image** - 生成纯色图片（IMAGE）

### 🔧 核心功能
- 支持多种视频格式（MP4, WebM, MKV, AVI）
- 智能 GIF 压缩算法，自动优化文件大小
- 可配置的帧率、尺寸和颜色数量
- 外部预览支持（浏览器/媒体播放器）
- 自动文件管理（复制到输出目录）

## 🚀 安装

### 方法一：通过 ComfyUI Manager（推荐）
1. 打开 ComfyUI Manager
2. 搜索 "ComfyUI Video Utilities"
3. 点击安装

### 方法二：手动安装
1. 克隆此仓库到 ComfyUI 的 custom_nodes 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI_Video_Utilities.git
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 重启 ComfyUI

## 📦 依赖要求

### 必需依赖
- Python 3.8+
- ComfyUI
- ffmpeg (系统级安装)
- Pillow (PIL)
- numpy
- torch
- ffmpeg-python>=0.2.0
- imageio-ffmpeg>=0.4.9

### 可选依赖
- opencv-python (用于高级视频处理)
- 人脸修复功能依赖 (见 requirements_face_restore.txt)

### 安装依赖
```bash
# 基础功能
pip install -r requirements.txt

# 人脸修复功能（可选）
pip install -r requirements_face_restore.txt
```

## 🎯 节点详解

### 1. Video Stitching
视频拼接节点，将多个视频文件合并为一个。

**功能说明：**
- 支持多种视频格式
- 自动处理不同分辨率和帧率
- 智能过渡效果

### 2. Get Last Frame
提取视频的最后一帧作为静态图像。

**功能说明：**
- 快速提取视频结尾帧
- 支持多种视频格式
- 输出高质量静态图像

### 3. Get VHS File Path
从 Video Combine 节点输出中提取视频文件路径。

**输入参数：**
- `filenames` (VHS_FILENAMES) - 来自 Video Combine 节点的文件名列表
- `sleep` (INT) - 等待时间（秒），默认 3 秒

**输出：**
- `AFVIDEO` (VIDEO) - 提取的视频文件路径

**功能说明：**
- 自动查找最新的视频文件
- 支持多种视频格式
- 可选择复制到输出目录
- 内置错误处理和重试机制

### 4. Video to GIF
将视频文件转换为 GIF 动画。

**输入参数：**
- `video` (VIDEO) - 输入视频文件
- `fps` (INT) - 输出帧率，默认 10
- `max_width` (INT) - 最大宽度，默认 512
- `max_size_mb` (FLOAT) - 最大文件大小（MB），默认 10.0
- `colors` (INT) - 颜色数量，默认 256
- `dither` (BOOLEAN) - 是否使用抖动，默认 True
- `loop` (INT) - 循环次数，0 表示无限循环，默认 0
- `keep_aspect` (BOOLEAN) - 保持宽高比，默认 True

**输出：**
- `GIF_PATH` (STRING) - 生成的 GIF 文件路径
- `STATUS` (STRING) - 转换状态信息

**功能说明：**
- 智能压缩算法，自动优化文件大小
- 支持两阶段压缩（如果文件过大）
- 可配置的颜色调色板
- 保持原始宽高比
- 详细的转换状态反馈

### 5. Video Preview
在 ComfyUI 界面中预览视频文件。

**输入参数：**
- `video` (VIDEO) - 输入视频文件

**输出：**
- UI 视频预览窗口

**功能说明：**
- 在 ComfyUI 界面中直接预览视频
- 自动清除之前的预览内容
- 支持视频控制（播放/暂停/进度条）
- 与 ComfyUI 界面完美集成

### 6. Upload Live Video
上传和加载视频文件，支持从 input/output 目录选择。

**输入参数：**
- `video` (SELECT) - 视频文件选择器，自动扫描 input 和 output 目录

**输出：**
- `IMAGE` (IMAGE) - 视频帧序列（N, H, W, C 格式）
- `AUDIO` (AUDIO) - 提取的音频数据
- `FILENAME` (STRING) - 视频文件名
- `FILE_AGE` (INT) - 文件年龄（秒）
- `STATUS` (STRING) - 加载状态信息

**功能说明：**
- 自动扫描 input 和 output 目录中的视频文件
- 支持多种视频格式（MP4, WebM, MKV, AVI）
- 按修改时间排序，最新的文件在前
- 提取所有视频帧作为图像序列
- 同时提取音频数据
- 提供详细的文件信息和状态反馈
- 支持动态刷新文件列表

### 7. Preview GIF
在外部应用程序中预览 GIF 文件。

**输入参数：**
- `gif_path` (STRING) - GIF 文件路径
- `copy_to_output` (BOOLEAN) - 是否复制到输出目录，默认 True
- `preview_method` (SELECT) - 预览方式：
  - `browser` - 在浏览器中打开（默认）
  - `media_player` - 在媒体播放器中打开

**输出：**
- `file_path` (STRING) - 最终文件路径
- UI 文本显示预览状态

**功能说明：**
- 支持浏览器和媒体播放器预览
- 自动文件路径管理
- 跨平台兼容（Windows/macOS/Linux）
- 详细的预览状态反馈

## 🎨 使用示例

### 基本工作流
```
[Upload Live Video] → [Video to GIF] → [Preview GIF]
```

### 视频预览工作流
```
[Upload Live Video] → [Video Preview]
```

### 高级工作流
```
[Video Combine] → [Get VHS File Path] → [Video to GIF] → [Preview GIF]
```

### 完整视频处理工作流
```
[Upload Live Video] → [Video Stitching] → [Video to GIF] → [Preview GIF]
```

### Live 监控工作流
```
[Live Video Monitor] → [展示/推理节点] → [保存/后处理]
```

### 参数配置示例
- **高质量 GIF**：fps=15, max_width=800, colors=256, dither=True
- **小文件 GIF**：fps=8, max_width=400, colors=128, dither=False
- **快速预览**：fps=5, max_width=300, colors=64

## ⚙️ 配置选项

### 环境变量
- `COMFYUI_VIDEO_UTILITIES_DEBUG` - 启用调试模式
- `COMFYUI_VIDEO_UTILITIES_TEMP_DIR` - 自定义临时目录

### 配置文件
创建 `config.json` 文件来自定义默认设置：
```json
{
  "default_fps": 10,
  "default_max_width": 512,
  "default_max_size_mb": 10.0,
  "default_colors": 256,
  "enable_dither": true,
  "preview_method": "browser"
}
```

## 🔧 故障排除

### 常见问题

**Q: ffmpeg 未找到**
A: 确保 ffmpeg 已安装并在系统 PATH 中，或设置环境变量指向 ffmpeg 可执行文件。

**Q: GIF 文件过大**
A: 尝试降低 fps、max_width 或 colors 参数，或启用两阶段压缩。

**Q: 预览无法打开**
A: 检查文件路径是否正确，确保文件存在且可访问。

**Q: 内存不足**
A: 降低视频分辨率或帧率，或使用更小的 max_width 值。

### 调试模式
启用调试模式以获取详细的日志信息：
```bash
export COMFYUI_VIDEO_UTILITIES_DEBUG=1
```

## 📊 性能优化

### 推荐设置
- **快速转换**：fps=8, max_width=400
- **平衡质量**：fps=12, max_width=600
- **最高质量**：fps=15, max_width=800

### 内存使用
- 大视频文件建议降低分辨率
- 使用较小的颜色调色板可减少内存使用
- 启用两阶段压缩可处理大文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发环境设置
1. Fork 此仓库
2. 创建功能分支
3. 安装开发依赖：`pip install -r requirements-dev.txt`
4. 运行测试：`python -m pytest tests/`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- ComfyUI 社区
- UtilNodes-ComfyUI 项目（Get VHS File Path 节点参考）
- FFmpeg 项目
- 所有贡献者和用户

## 📁 项目结构

```
ComfyUI_Video_Utilities/
├── __init__.py                 # 扩展入口文件
├── Video_Utilities.py          # 主要节点实现
├── requirements.txt            # 基础依赖
├── requirements_face_restore.txt  # 人脸修复依赖
├── README.md                   # 项目说明文档
├── LICENSE                     # 许可证文件
├── js/                         # 前端 JavaScript 文件
│   └── video_preview.js        # 视频预览前端代码
├── examples/                   # 示例工作流
│   ├── SeedReam4API_example.json
│   └── ...
└── 其他工具文件...
```

## 🎨 示例工作流

项目包含多个示例工作流文件，位于 `examples/` 目录中：

## 📞 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 搜索 [Issues](https://github.com/your-username/ComfyUI_Video_Utilities/issues)
3. 创建新的 Issue
4. 联系维护者

## 🔄 更新日志

### v1.2.0 (2025-09-26)
- ✨ 新增 Live_Video_Monitor（含前端预览、自动刷新）
- ✨ 新增 Load_AF_Video（含前端预览与上传控件）
- ✨ 新增 Prompt_Text_Node（通用 STRING 输出）
- ✨ 新增 RGB_Empty_Image 纯色图像节点
- 🔄 统一 Upload_Live_Video 与 Live_Video_Monitor 的 `FILENAMES` 输出为 `VHS_FILENAMES`（与 Video Combine 兼容）
- 🛠 修复/增强：视频预览脚本、WEB_DIRECTORY、类型兼容

### v1.1.0 (2025-01-26)
- ✨ 添加 Video Preview 节点 - 在 ComfyUI 界面中预览视频
- ✨ 添加 Upload Live Video 节点 - 上传和加载视频文件
- ✨ 添加前端 JavaScript 支持 - video_preview.js
- 🔧 完善视频处理功能
- 📚 更新文档和示例

### v1.0.0 (2025-01-26)
- ✨ 初始版本发布
- ✨ 添加 Get VHS File Path 节点
- ✨ 添加 Video to GIF 节点
- ✨ 添加 Preview GIF 节点
- ✨ 添加 Video Stitching 节点
- ✨ 添加 Get Last Frame 节点
- 🔧 移除 UI 预览功能，专注于外部预览
- 📚 完善文档和示例

---

**作者**: Ken-Chen  
**版本**: 1.2.0  
**最后更新**: 2025-09-26

---
## 💝 Support the Project

<div align="center">

### ☕ Buy Me a Coffee

If you find IndexTTS2 helpful and it has made your voice synthesis projects easier, consider supporting the development!

**🎯 Your support helps:**
- 🚀 Accelerate new feature development
- 🧠 Enhance AI capabilities
- 🔧 Improve system stability
- 📚 Create better documentation
- 🌍 Support the open-source community

</div>

<table>
<tr>
<td width="50%" align="center">

**💬 WeChat Contact**

<img src="https://github.com/xuchenxu168/images/blob/main/%E5%BE%AE%E4%BF%A1%E5%8F%B7.jpg" alt="WeChat QR Code" width="200" height="200">

*Scan to add WeChat*
*扫码添加微信*

**WeChat ID**: `Kenchen7168`

</td>
<td width="50%" align="center">

**☕ Support Development**

<img src="https://github.com/xuchenxu168/images/blob/main/%E6%94%B6%E6%AC%BE%E7%A0%81.jpg" width="200" height="200">

*Scan to buy me a coffee*
*扫码请我喝咖啡*

**💝 Every coffee counts!**
*每一杯咖啡都是支持！*

</td>
</tr>
</table>

<div align="center">

**🙏 Thank you for your support!**

*Your contributions, whether through code, feedback, or coffee, make IndexTTS2 better for everyone!*

**谢谢您的支持！无论是代码贡献、反馈建议还是请我喝咖啡，都让IndexTTS2变得更好！**

</div>

---

<div align="center">

### 🚀 Ready to Create Amazing AI-Enhanced Voice Content?

**[⬆️ Back to Top](#-comfyui-indextts2-plugin)** • **[📦 Install Now](#-installation)** • **[🎯 Quick Start](#-quick-start)** • **[🧠 AI Features](#-ai-enhancement-features)** • **[🤝 Join Community](#-community--contributing)** • **[💝 Support Project](#-support-the-project)**

---

**🎙️ IndexTTS2 ComfyUI Plugin** - *Revolutionary AI-Enhanced Voice Synthesis Platform*

**🧠 Now with Advanced AI Enhancement Systems** - *Intelligent, Self-Learning, Continuously Improving*

![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)
![AI Enhanced](https://img.shields.io/badge/AI%20Enhanced-🧠-purple?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-💚-green?style=for-the-badge)
![Community Driven](https://img.shields.io/badge/Community%20Driven-🤝-blue?style=for-the-badge)

</div>

