"""
Text To Video Dynamic Node
动态字幕节点（逐词显示，抖音风格）
"""

import os
import sys
import tempfile
import time
import random
import cv2

from ..utils.subtitle_renderer import SubtitleRenderer


def _log_info(message):
    print(f"[Text_To_Video_Dynamic] {message}")


def _log_error(message):
    print(f"[Text_To_Video_Dynamic ERROR] {message}")


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


class VideoUtilitiesTextToVideoDynamic:
    """动态字幕节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        font_files = get_font_list()

        return {
            "required": {
                "video": ("VIDEO",),
                "subtitles": ("STRING", {"forceInput": True}),  # 逐词时间戳
                "font_file": (font_files, {"default": font_files[0]}),
                "font_size": ("INT", {"default": 24, "min": 10, "max": 100, "step": 1}),
                "font_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "yellow"}),
                "text_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "position": (["右上", "右中", "右下", "中上", "正中", "中下", "左上", "左中", "左下"], {"default": "中下"}),
                "background": (["yes", "no"], {"default": "yes"}),
            },
            "optional": {
                "stroke_width": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "stroke_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "black"}),
                "max_lines": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "clearance_threshold": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "subtitle_extend_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "延长字幕显示时间（秒），解决歌手还没唱完字幕就消失的问题"}),
                "display_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "字幕显示速度倍数（1.0=正常，2.0=2倍速，0.5=0.5倍速）"}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "X轴偏移量（正数向右，负数向左）"}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "Y轴偏移量（正数向上，负数向下）"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "add_dynamic_subtitles"
    CATEGORY = "Video_Utilities/Subtitle"
    
    def add_dynamic_subtitles(
        self,
        video,
        subtitles,
        font_file,
        font_size,
        font_color,
        text_direction,
        position,
        background,
        stroke_width=3.0,
        stroke_color="black",
        max_lines=3,
        clearance_threshold=2.0,
        subtitle_extend_time=0.0,
        display_speed=1.0,
        offset_x=0,
        offset_y=0
    ):
        """
        添加动态字幕（逐词显示）

        Args:
            video: 输入视频
            subtitles: 逐词时间戳字符串
            其他参数: 字幕样式参数

        Returns:
            带字幕的视频
        """
        try:
            _log_info("🎯 开始添加动态字幕...")

            # 1. 提取视频路径
            video_path = self._extract_video_path(video)
            if not video_path:
                _log_error("❌ 无法提取视频路径")
                return (video,)

            _log_info(f"📹 输入视频: {video_path}")

            # 2. 解析逐词时间戳
            words_list = self._parse_word_timestamps(subtitles)
            if not words_list:
                _log_error("❌ 逐词时间戳为空或格式错误")
                _log_info("💡 提示: 请使用 faster-whisper 引擎获取逐词时间戳")
                return (video,)

            _log_info(f"📝 解析到 {len(words_list)} 个词")

            # 2.5. 调整显示速度
            if display_speed != 1.0:
                _log_info(f"⚡ 调整显示速度: {display_speed}x")
                words_list = self._adjust_display_speed(words_list, display_speed)

            # 3. 获取字体路径
            fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Fonts")
            font_path = os.path.join(fonts_dir, font_file)

            if not os.path.exists(font_path):
                _log_error(f"❌ 字体文件不存在: {font_path}")
                return (video,)

            # 4. 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _log_error(f"❌ 无法打开视频: {video_path}")
                return (video,)

            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            _log_info(f"📐 视频尺寸: {video_width}x{video_height}, FPS: {fps}")

            # 5. 生成输出路径（保存到 ComfyUI output 目录）
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                # 如果无法获取 output 目录，使用临时目录
                output_dir = tempfile.gettempdir()

            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            output_filename = f"dynamic_subtitle_{timestamp}_{random_suffix}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # 6. 渲染动态字幕
            renderer = SubtitleRenderer(video_width, video_height, fps)
            renderer.render_dynamic_subtitles(
                video_path=video_path,
                output_path=output_path,
                words_list=words_list,
                font_path=font_path,
                font_size=font_size,
                font_color=font_color,
                text_direction=text_direction,
                position=position,
                background=background,
                max_lines=max_lines,
                clearance_threshold=clearance_threshold,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                subtitle_extend_time=subtitle_extend_time,
                offset_x=offset_x,
                offset_y=offset_y
            )

            _log_info(f"✅ 动态字幕添加完成: {output_path}")

            # 7. 返回视频路径（字符串格式）
            return (output_path,)

        except Exception as e:
            _log_error(f"❌ 动态字幕添加失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return (video,)

    def _extract_video_path(self, video):
        """提取视频路径"""
        if isinstance(video, str):
            return video
        elif hasattr(video, 'saved_path'):
            return video.saved_path
        elif isinstance(video, dict):
            if 'filename' in video:
                from folder_paths import get_output_directory
                output_dir = get_output_directory()
                subfolder = video.get('subfolder', '')
                filename = video['filename']
                return os.path.join(output_dir, subfolder, filename)
        return None
    
    def _parse_word_timestamps(self, subtitles):
        """
        解析逐词时间戳
        
        格式: (start, end) word
        例如: (0.0, 0.5) 这
        
        Returns:
            [(start, end, word), ...]
        """
        words_list = []
        
        for line in subtitles.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: (start, end) word
            if line.startswith('(') and ')' in line:
                try:
                    # 提取时间戳
                    timestamp_end = line.index(')')
                    timestamp_str = line[1:timestamp_end]
                    word = line[timestamp_end + 1:].strip()
                    
                    # 解析时间
                    parts = timestamp_str.split(',')
                    if len(parts) == 2:
                        start = float(parts[0].strip())
                        end = float(parts[1].strip())
                        words_list.append((start, end, word))
                except Exception as e:
                    _log_error(f"解析字幕行失败: {line}, 错误: {e}")
                    continue

        return words_list

    def _adjust_display_speed(self, words_list, speed):
        """
        调整字幕显示速度

        Args:
            words_list: 原始逐词时间戳 [(start, end, word), ...]
            speed: 速度倍数（1.0=正常，2.0=2倍速，0.5=0.5倍速）

        Returns:
            调整后的逐词时间戳 [(start, end, word), ...]

        工作原理：
        - 速度 > 1.0：字幕显示更快（缩短每个词的持续时间）
        - 速度 < 1.0：字幕显示更慢（延长每个词的持续时间）
        - 保持词与词之间的相对时间关系
        """
        if not words_list or speed <= 0:
            return words_list

        adjusted_words = []
        time_offset = 0.0  # 累积的时间偏移

        for i, (start, end, word) in enumerate(words_list):
            # 计算原始持续时间
            original_duration = end - start

            # 调整持续时间（除以速度倍数）
            adjusted_duration = original_duration / speed

            # 计算新的开始和结束时间
            if i == 0:
                # 第一个词：保持原始开始时间
                new_start = start
            else:
                # 后续词：使用上一个词的结束时间
                new_start = adjusted_words[-1][1]

            new_end = new_start + adjusted_duration

            adjusted_words.append((round(new_start, 2), round(new_end, 2), word))

        return adjusted_words

