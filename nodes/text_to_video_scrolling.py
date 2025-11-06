"""
Text To Video Scrolling Node
专业滚动字幕节点 - 电影级滚动效果
"""

import os
import sys
import tempfile
import time
import random
import cv2

from ..utils.scrolling_renderer import ScrollingRenderer


def _log_info(message):
    print(f"[Text_To_Video_Scrolling] {message}")


def _log_error(message):
    print(f"[Text_To_Video_Scrolling ERROR] {message}")


def extract_video_path(video):
    """提取视频路径（兼容多种 VIDEO 格式）"""
    if isinstance(video, str):
        return video
    elif isinstance(video, dict):
        if 'filename' in video:
            import folder_paths
            video_dir = folder_paths.get_input_directory()
            return os.path.join(video_dir, video['filename'])
        elif 'saved_path' in video:
            return video['saved_path']
    elif hasattr(video, 'saved_path'):
        return video.saved_path
    return None


def video_to_comfyui_video(video_path):
    """将视频路径转换为 ComfyUI VIDEO 对象"""
    if not video_path or not os.path.exists(video_path):
        return None
    return video_path


def get_font_list():
    """获取 Fonts 目录下的字体文件列表"""
    fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Fonts")
    font_files = []
    
    if os.path.exists(fonts_dir):
        for file in sorted(os.listdir(fonts_dir)):
            if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                font_files.append(file)
    
    if not font_files:
        font_files = ["请将字体文件放入 Fonts 目录"]
    
    return font_files


class VideoUtilitiesTextToVideoScrolling:
    """专业滚动字幕节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        font_files = get_font_list()

        return {
            "required": {
                "video": ("VIDEO",),
                "text": ("STRING", {"multiline": True, "default": "在这里输入滚动字幕文本\n支持多行文本\n每行会自动换行"}),
                "font_file": (font_files, {"default": font_files[0]}),
                "font_size": ("INT", {"default": 36, "min": 12, "max": 120, "step": 1}),
                "font_color": (["white", "yellow", "black", "red", "green", "blue", "cyan", "magenta"], {"default": "yellow"}),
                "scroll_type": ([
                    "vertical_up",              # 垂直向上滚动（片尾字幕）
                    "vertical_down",            # 垂直向下滚动
                    "horizontal_left_top",      # 水平向左滚动（顶部）
                    "horizontal_left_center",   # 水平向左滚动（中部）
                    "horizontal_left_bottom",   # 水平向左滚动（底部）
                    "horizontal_right_top",     # 水平向右滚动（顶部）
                    "horizontal_right_center",  # 水平向右滚动（中部）
                    "horizontal_right_bottom",  # 水平向右滚动（底部）
                    "star_wars",                # 星战式3D透视滚动
                    "fade_scroll"               # 渐变滚动
                ], {"default": "vertical_up"}),
                "scroll_speed": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 200.0, "step": 5.0, "tooltip": "滚动速度（像素/秒）"}),
                "start_position": ([
                    "bottom",           # 从底部开始
                    "top",              # 从顶部开始
                    "center",           # 从中心开始
                    "off_screen"        # 从屏幕外开始
                ], {"default": "off_screen"}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "循环滚动（滚动完成后从头开始）"}),
            },
            "optional": {
                "background_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "背景不透明度"}),
                "stroke_width": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "stroke_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "black"}),
                "fade_in_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "淡入时长（秒）"}),
                "fade_out_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "淡出时长（秒）"}),
                "line_spacing": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "行间距倍数"}),
                "text_align": (["left", "center", "right"], {"default": "center"}),
                "perspective_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "3D透视强度（仅星战式）"}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "X轴偏移量（正数向右，负数向左）"}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "Y轴偏移量（正数向上，负数向下）"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "add_scrolling_text"
    CATEGORY = "Video_Utilities/Subtitle"
    
    def add_scrolling_text(
        self,
        video,
        text,
        font_file,
        font_size,
        font_color,
        scroll_type,
        scroll_speed,
        start_position,
        loop=True,
        background_opacity=0.0,
        stroke_width=2.0,
        stroke_color="black",
        fade_in_duration=1.0,
        fade_out_duration=1.0,
        line_spacing=1.5,
        text_align="center",
        perspective_strength=0.5,
        offset_x=0,
        offset_y=0
    ):
        """
        添加滚动字幕

        Args:
            video: 输入视频
            text: 滚动文本
            其他参数: 字幕样式参数

        Returns:
            带滚动字幕的视频
        """
        try:
            # 1. 提取视频路径
            video_path = extract_video_path(video)
            if not video_path or not os.path.exists(video_path):
                _log_error("❌ 无法获取视频路径或视频文件不存在")
                return (video,)

            _log_info(f"🎬 开始处理视频: {video_path}")

            # 2. 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _log_error("❌ 无法打开视频")
                return (video,)

            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()

            _log_info(f"📹 视频信息: {video_width}x{video_height} @ {fps}fps, 时长: {duration:.2f}秒")

            # 3. 获取字体路径
            fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Fonts")
            font_path = os.path.join(fonts_dir, font_file)

            if not os.path.exists(font_path):
                _log_error(f"❌ 字体文件不存在: {font_path}")
                return (video,)

            # 4. 生成输出路径
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                output_dir = tempfile.gettempdir()

            output_filename = f"scrolling_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # 5. 渲染滚动字幕
            _log_info(f"🎨 开始渲染滚动字幕（类型: {scroll_type}）...")
            renderer = ScrollingRenderer(video_width, video_height, fps)

            renderer.render_scrolling_text(
                video_path,
                output_path,
                text,
                font_path,
                font_size,
                font_color,
                scroll_type,
                scroll_speed,
                start_position,
                loop,
                background_opacity,
                stroke_width,
                stroke_color,
                fade_in_duration,
                fade_out_duration,
                line_spacing,
                text_align,
                perspective_strength,
                offset_x,
                offset_y
            )

            _log_info(f"✅ 滚动字幕渲染完成: {output_path}")

            # 6. 转换为 ComfyUI VIDEO 对象
            output_video = video_to_comfyui_video(output_path)
            if not output_video:
                _log_error("❌ 视频对象转换失败")
                return (video,)

            return (output_video,)

        except Exception as e:
            _log_error(f"❌ 滚动字幕添加失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return (video,)