# Video Utilities Plugin for ComfyUI
# 专注于视频处理和工具功能
# 作者: Ken-Chen
# 版本: 2.0.0

import importlib

# 合并所有节点的映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Delay module import to avoid startup import errors
def load_modules():
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # Import server routes first
    try:
        # 使用不同的名称避免与 ComfyUI 的 server 模块冲突
        from . import server as video_server
        print("[Video Utilities] Server routes loaded")
    except Exception as e:
        print(f"[Video Utilities] Warning: Could not load server routes: {e}")
        import traceback
        traceback.print_exc()

    # Import Video Utilities module
    try:
        # Try relative import
        try:
            from . import Video_Utilities as video_utils
        except (ImportError, ValueError):
            # If relative import fails, try absolute import
            video_utils = importlib.import_module('Video_Utilities')

        if hasattr(video_utils, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(video_utils.NODE_CLASS_MAPPINGS)
        if hasattr(video_utils, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(video_utils.NODE_DISPLAY_NAME_MAPPINGS)
        print("[Video Utilities] Module loaded successfully")
    except Exception as e:
        print(f"[Video Utilities] Module loading failed: {e}")

    # Import new nodes
    try:
        from .nodes import (
            VideoUtilitiesAudioToText,
            VideoUtilitiesTextToVideoStatic,
            VideoUtilitiesTextToVideoDynamic,
            VideoUtilitiesTextToVideoScrolling,
            VideoUtilitiesColorPicker,
        )

        NODE_CLASS_MAPPINGS.update({
            "Audio_To_Text": VideoUtilitiesAudioToText,
            "Text_To_Video_Static": VideoUtilitiesTextToVideoStatic,
            "Text_To_Video_Dynamic": VideoUtilitiesTextToVideoDynamic,
            "Text_To_Video_Scrolling": VideoUtilitiesTextToVideoScrolling,
            "Color_Picker": VideoUtilitiesColorPicker,
        })

        NODE_DISPLAY_NAME_MAPPINGS.update({
            "Audio_To_Text": "Audio To Text 🎤",
            "Text_To_Video_Static": "Text To Video (Static) 📝",
            "Text_To_Video_Dynamic": "Text To Video (Dynamic) 🎯",
            "Text_To_Video_Scrolling": "Text To Video (Scrolling) 🎬",
            "Color_Picker": "Color Picker 🎨",
        })

        print("[Video Utilities] New nodes loaded successfully")
    except Exception as e:
        print(f"[Video Utilities] New nodes loading failed: {e}")
        import traceback
        traceback.print_exc()

# Load modules immediately
load_modules()


# 设置 Web 目录 - 必须在 NODE_CLASS_MAPPINGS 之前设置
WEB_DIRECTORY = "./js"

# ComfyUI Registry ID
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[Video Utilities] Plugin loading completed!")
print(f"[Video Utilities] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
print(f"[Video Utilities] Node list: {list(NODE_CLASS_MAPPINGS.keys())}")