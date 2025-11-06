"""
Text To Video Static Node
静态字幕节点（保留所有 11 种动画特效）
"""

import os
import sys
import tempfile
import time
import random
import cv2

from ..utils.subtitle_renderer import SubtitleRenderer
from ..utils.animation import get_animation_list


def _log_info(message):
    print(f"[Text_To_Video_Static] {message}")


def _log_error(message):
    print(f"[Text_To_Video_Static ERROR] {message}")


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


class VideoUtilitiesTextToVideoStatic:
    """静态字幕节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        font_files = get_font_list()
        animation_list = get_animation_list()

        return {
            "required": {
                "video": ("VIDEO",),
                "subtitles": ("STRING", {"forceInput": True}),  # 逐句时间戳
                "font_file": (font_files, {"default": font_files[0]}),
                "font_size": ("INT", {"default": 24, "min": 10, "max": 100, "step": 1}),
                "font_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "yellow"}),
                "text_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "position": (["右上", "右中", "右下", "中上", "正中", "中下", "左上", "左中", "左下"], {"default": "中下"}),
                "background": (["yes", "no"], {"default": "yes"}),
                "animation": (animation_list, {"default": "fade_in"}),
                "animation_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "stroke_width": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "stroke_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "black"}),
                "subtitle_extend_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "延长字幕显示时间（秒），解决歌手还没唱完字幕就消失的问题"}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "X轴偏移量（正数向右，负数向左）"}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "Y轴偏移量（正数向上，负数向下）"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "add_static_subtitles"
    CATEGORY = "Video_Utilities/Subtitle"
    
    def add_static_subtitles(
        self,
        video,
        subtitles,
        font_file,
        font_size,
        font_color,
        text_direction,
        position,
        background,
        animation,
        animation_duration,
        stroke_width=2.0,
        stroke_color="black",
        subtitle_extend_time=0.0,
        offset_x=0,
        offset_y=0
    ):
        """
        添加静态字幕

        Args:
            video: 输入视频
            subtitles: 逐句时间戳字符串
            其他参数: 字幕样式参数

        Returns:
            带字幕的视频
        """
        try:
            # 1. 提取视频路径
            video_path = extract_video_path(video)
            if not video_path or not os.path.exists(video_path):
                _log_error("❌ 无法获取视频路径或视频文件不存在")
                return (video,)

            _log_info(f"🎬 开始处理视频: {video_path}")

            # 2. 解析字幕时间戳
            sentences_list = self._parse_subtitles(subtitles)
            if not sentences_list:
                _log_error("❌ 字幕解析失败或为空")
                return (video,)

            _log_info(f"📝 解析了 {len(sentences_list)} 个字幕")

            # 3. 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _log_error("❌ 无法打开视频")
                return (video,)

            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            _log_info(f"📹 视频信息: {video_width}x{video_height} @ {fps}fps")

            # 4. 获取字体路径
            fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Fonts")
            font_path = os.path.join(fonts_dir, font_file)

            if not os.path.exists(font_path):
                _log_error(f"❌ 字体文件不存在: {font_path}")
                return (video,)

            # 5. 生成输出路径（保存到 ComfyUI output 目录）
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                # 如果无法获取 output 目录，使用临时目录
                output_dir = tempfile.gettempdir()

            output_filename = f"subtitle_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # 6. 渲染字幕
            _log_info(f"🎨 开始渲染字幕（动画: {animation}, 方向: {text_direction}）...")
            renderer = SubtitleRenderer(video_width, video_height, fps)

            renderer.render_static_subtitles(
                video_path,
                output_path,
                sentences_list,
                font_path,
                font_size,
                font_color,
                text_direction,
                position,
                background,
                animation,
                animation_duration,
                stroke_width,
                stroke_color,
                subtitle_extend_time,
                offset_x,
                offset_y
            )

            _log_info(f"✅ 字幕渲染完成: {output_path}")

            # 7. 转换为 ComfyUI VIDEO 对象
            output_video = video_to_comfyui_video(output_path)
            if not output_video:
                _log_error("❌ 视频对象转换失败")
                return (video,)

            return (output_video,)

        except Exception as e:
            _log_error(f"❌ 字幕添加失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return (video,)
    
    def _parse_subtitles(self, subtitles):
        """
        解析字幕时间戳

        格式: (start, end) text
        例如: (0.0, 3.0) 这是一段测试文本

        支持两种输入：
        1. 逐句时间戳：每行是一个完整的句子
        2. 逐词时间戳：每行是一个词，需要合并为句子

        Returns:
            [(start, end, text), ...]
        """
        sentences_list = []

        for line in subtitles.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # 解析格式: (start, end) text
            if line.startswith('(') and ')' in line:
                try:
                    # 提取时间戳
                    timestamp_end = line.index(')')
                    timestamp_str = line[1:timestamp_end]
                    text = line[timestamp_end + 1:].strip()

                    # 解析时间
                    parts = timestamp_str.split(',')
                    if len(parts) == 2:
                        start = float(parts[0].strip())
                        end = float(parts[1].strip())
                        sentences_list.append((start, end, text))
                except Exception as e:
                    _log_error(f"解析字幕行失败: {line}, 错误: {e}")
                    continue

        # 检测是否是逐词时间戳（需要合并为句子）
        if self._is_word_timestamps(sentences_list):
            _log_info("🔍 检测到逐词时间戳，正在合并为句子...")
            sentences_list = self._merge_words_to_sentences(sentences_list)
            _log_info(f"✅ 合并完成，得到 {len(sentences_list)} 个句子")

        return sentences_list

    def _is_word_timestamps(self, items_list):
        """
        检测是否是逐词时间戳

        判断标准：
        1. 如果平均每个文本长度 <= 3 个字符，可能是逐词时间戳
        2. 如果有 <NEWLINE> 标记，肯定是逐词时间戳

        Returns:
            True: 逐词时间戳
            False: 逐句时间戳
        """
        if not items_list:
            return False

        # 检查是否有 <NEWLINE> 标记
        for _, _, text in items_list:
            if text == "<NEWLINE>":
                return True

        # 计算平均文本长度
        total_length = sum(len(text) for _, _, text in items_list)
        avg_length = total_length / len(items_list)

        # 如果平均长度 <= 3，认为是逐词时间戳
        return avg_length <= 3

    def _merge_words_to_sentences(self, words_list):
        """
        将逐词时间戳合并为逐句时间戳

        合并规则：
        1. 遇到 <NEWLINE> 标记时，结束当前句子
        2. 遇到较长的静音间隔（> 1.5 秒）时，结束当前句子

        Args:
            words_list: [(start, end, word), ...]

        Returns:
            [(start, end, sentence), ...]
        """
        sentences_list = []
        current_text = ""
        sentence_start = 0.0
        last_word_end = 0.0
        silence_threshold = 1.5  # 静音阈值（秒）

        for i, (start, end, word) in enumerate(words_list):
            # 检查是否是换行符标记
            if word == "<NEWLINE>":
                # 结束当前句子
                if current_text:
                    sentences_list.append((sentence_start, last_word_end, current_text))
                    current_text = ""
                continue

            # 检查是否需要开始新句子（静音超过阈值）
            if current_text and (start - last_word_end) > silence_threshold:
                # 保存当前句子
                sentences_list.append((sentence_start, last_word_end, current_text))
                current_text = ""

            # 累积文字
            if not current_text:
                current_text = word
                sentence_start = start
            else:
                current_text += word

            last_word_end = end

        # 保存最后一个句子
        if current_text:
            sentences_list.append((sentence_start, last_word_end, current_text))

        return sentences_list

