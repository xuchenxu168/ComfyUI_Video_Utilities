# ComfyUI Video Utilities - New Nodes
# 新节点模块

from .audio_to_text import VideoUtilitiesAudioToText
from .text_to_video_static import VideoUtilitiesTextToVideoStatic
from .text_to_video_dynamic import VideoUtilitiesTextToVideoDynamic
from .text_to_video_scrolling import VideoUtilitiesTextToVideoScrolling
from .color_picker import VideoUtilitiesColorPicker

__all__ = [
    'VideoUtilitiesAudioToText',
    'VideoUtilitiesTextToVideoStatic',
    'VideoUtilitiesTextToVideoDynamic',
    'VideoUtilitiesTextToVideoScrolling',
    'VideoUtilitiesColorPicker',
]

