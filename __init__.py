# Doubao-Seed Plugin for ComfyUI
# 专注于Doubao图像和视频生成功能
# 作者: Ken-Chen
# 版本: 2.0.0

import importlib

# 合并所有节点的映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Delay module import to avoid startup import errors
def load_modules():
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

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

# Load modules immediately
load_modules()


# 设置 Web 目录 - 必须在 NODE_CLASS_MAPPINGS 之前设置
WEB_DIRECTORY = "./js"

print("[Video Utilities] Plugin loading completed!")
print(f"[Video Utilities] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
print(f"[Video Utilities] Node list: {list(NODE_CLASS_MAPPINGS.keys())}")