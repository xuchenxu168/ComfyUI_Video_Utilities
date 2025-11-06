"""
字幕渲染器
支持静态和动态字幕，保留所有现有功能
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from .animation import apply_animation
from .text_wrapper import smart_wrap_text, remove_punctuation


def _log_info(message):
    print(f"[SubtitleRenderer] {message}")


def _log_error(message):
    print(f"[SubtitleRenderer ERROR] {message}")


class SubtitleRenderer:
    """字幕渲染器"""
    
    def __init__(self, video_width: int, video_height: int, fps: float):
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
    
    def render_static_subtitles(
        self,
        video_path: str,
        output_path: str,
        sentences_list: List[Tuple[float, float, str]],
        font_path: str,
        font_size: int,
        font_color: str,
        text_direction: str,
        position: str,
        background: str,
        animation: str,
        animation_duration: float,
        stroke_width: float = 2.0,
        stroke_color: str = "black",
        subtitle_extend_time: float = 0.0,
        offset_x: int = 0,
        offset_y: int = 0
    ):
        """
        渲染静态字幕

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            sentences_list: 句子列表 [(start, end, text), ...]
            其他参数: 字幕样式参数
        """
        _log_info(f"🎬 开始渲染静态字幕，共 {len(sentences_list)} 个句子")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 加载字体
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            _log_error(f"加载字体失败: {e}")
            font = ImageFont.load_default()

        # 颜色映射
        color_map = {
            "white": (255, 255, 255),
            "yellow": (255, 255, 0),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }
        font_rgb = color_map.get(font_color, (255, 255, 0))
        stroke_rgb = color_map.get(stroke_color, (0, 0, 0)) if stroke_width > 0 else None

        # 背景设置
        bg_opacity = 0.5 if background == "yes" else 0.0
        bg_rgb = (0, 0, 0)  # 黑色背景
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.video_width, self.video_height))
        
        # 逐帧处理
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / self.fps
            
            # 查找当前时间的字幕
            current_subtitle = None
            subtitle_start = 0
            subtitle_end = 0
            
            for start, end, text in sentences_list:
                # 延长字幕显示时间
                extended_end = end + subtitle_extend_time
                if start <= current_time <= extended_end:
                    # 去除标点符号
                    import string
                    punctuation = string.punctuation + '，。！？；：""''（）《》、·…—'
                    current_subtitle = ''.join([c for c in text if c not in punctuation])
                    subtitle_start = start
                    subtitle_end = extended_end
                    break

            # 如果有字幕，绘制
            if current_subtitle and current_subtitle.strip():
                # 计算动画进度
                subtitle_duration = subtitle_end - subtitle_start
                subtitle_elapsed = current_time - subtitle_start
                subtitle_progress = subtitle_elapsed / subtitle_duration if subtitle_duration > 0 else 1.0

                # 动画进度（前 animation_duration 秒）
                if subtitle_elapsed < animation_duration:
                    animation_progress = subtitle_elapsed / animation_duration
                else:
                    animation_progress = 1.0

                # 绘制字幕
                frame = self._draw_subtitle_on_frame(
                    frame,
                    current_subtitle,
                    font,
                    font_rgb,
                    bg_rgb,
                    bg_opacity,
                    text_direction,
                    position,
                    stroke_width,
                    stroke_rgb,
                    animation,
                    animation_progress,
                    subtitle_progress,
                    offset_x,
                    offset_y
                )
            
            out.write(frame)
            frame_idx += 1
            
            # 进度显示
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                _log_info(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        cap.release()
        out.release()

        # 使用 ffmpeg 合并音频
        _log_info(f"🎵 正在合并音频...")
        self._merge_audio(video_path, output_path)

        _log_info(f"✅ 静态字幕渲染完成: {output_path}")
    
    def _draw_subtitle_on_frame(
        self,
        frame: np.ndarray,
        text: str,
        font: ImageFont.FreeTypeFont,
        font_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
        bg_opacity: float,
        text_direction: str,
        position: str,
        stroke_width: float,
        stroke_color: Optional[Tuple[int, int, int]],
        animation: str,
        animation_progress: float,
        subtitle_progress: float,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> np.ndarray:
        """在帧上绘制字幕"""

        # 转换为 PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')

        # 计算文字尺寸和位置
        wrapped_text = text  # 默认值
        if text_direction == "vertical":
            text_width, text_height = self._get_vertical_text_size(draw, text, font)
        else:
            # 智能换行
            max_width = int(self.video_width * 0.9)
            wrapped_text = smart_wrap_text(text, max_width, font.path, font.size, 'zh')
            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=3)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # 计算基础位置
        # 位置格式：右上、右中、右下、中上、正中、中下、左上、左中、左下
        # 兼容旧格式：bottom -> 中下, top -> 中上, middle -> 正中
        position_map = {
            "bottom": "中下",
            "top": "中上",
            "middle": "正中",
            "中中": "正中"  # 兼容旧的"中中"
        }
        position = position_map.get(position, position)

        # 解析位置
        if position in ["右上", "右中", "右下"]:
            # 右侧
            horizontal_offset = int(self.video_width * 0.05)
            base_x = self.video_width - text_width - horizontal_offset
        elif position in ["中上", "正中", "中下"]:
            # 中间
            base_x = (self.video_width - text_width) // 2
        else:  # 左上、左中、左下
            # 左侧
            horizontal_offset = int(self.video_width * 0.05)
            base_x = horizontal_offset

        if position in ["右上", "中上", "左上"]:
            # 上方
            vertical_offset = int(self.video_height * 0.05)
            base_y = vertical_offset
        elif position in ["右中", "正中", "左中"]:
            # 中间
            base_y = (self.video_height - text_height) // 2
        else:  # 右下、中下、左下
            # 下方
            vertical_offset = int(self.video_height * 0.05)
            base_y = self.video_height - text_height - vertical_offset

        # 应用偏移量
        # offset_x: 正数向右，负数向左
        # offset_y: 正数向上，负数向下（注意：图像坐标系 Y 轴向下，所以要取反）
        base_x += offset_x
        base_y -= offset_y

        # 应用动画效果
        x, y, alpha, scale, color = apply_animation(
            animation,
            animation_progress,
            subtitle_progress,
            base_x,
            base_y,
            font_color,
            self.video_width,
            self.video_height,
            text_height
        )

        # 应用缩放效果（如果需要）
        scaled_font = font
        scaled_text_width = text_width
        scaled_text_height = text_height

        if scale != 1.0 and scale > 0.1:  # 确保缩放比例不会太小
            # 创建缩放后的字体
            scaled_font_size = max(1, int(font.size * scale))  # 确保字体大小至少为1
            scaled_font = ImageFont.truetype(font.path, scaled_font_size)

            # 重新计算文字尺寸
            if text_direction == "vertical":
                scaled_text_width, scaled_text_height = self._get_vertical_text_size(draw, text, scaled_font)
            else:
                bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=scaled_font, spacing=3)
                scaled_text_width = bbox[2] - bbox[0]
                scaled_text_height = bbox[3] - bbox[1]

            # 调整位置以保持居中
            x = x + (text_width - scaled_text_width) // 2
            y = y + (text_height - scaled_text_height) // 2

        # 绘制背景
        if bg_opacity > 0:
            bg_padding = 10
            bg_x1 = x - bg_padding
            bg_y1 = y - bg_padding
            bg_x2 = x + scaled_text_width + bg_padding
            bg_y2 = y + scaled_text_height + bg_padding
            draw.rectangle(
                [bg_x1, bg_y1, bg_x2, bg_y2],
                fill=(*bg_color, int(255 * bg_opacity * alpha))
            )

        # 绘制文字
        if text_direction == "vertical":
            self._draw_vertical_text(draw, text, scaled_font, x, y, color, alpha, stroke_width, stroke_color)
        else:
            # 描边效果
            if stroke_width > 0 and stroke_color:
                for stroke_offset_x in range(-int(stroke_width), int(stroke_width) + 1):
                    for stroke_offset_y in range(-int(stroke_width), int(stroke_width) + 1):
                        if stroke_offset_x != 0 or stroke_offset_y != 0:
                            draw.multiline_text(
                                (x + stroke_offset_x, y + stroke_offset_y),
                                wrapped_text,
                                font=scaled_font,
                                fill=(*stroke_color, int(255 * alpha)),
                                spacing=3,
                                align="center"
                            )

            # 主文字
            draw.multiline_text(
                (x, y),
                wrapped_text,
                font=scaled_font,
                fill=(*color, int(255 * alpha)),
                spacing=3,
                align="center"
            )
        
        # 转换回 OpenCV 格式 (BGR)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _draw_vertical_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        alpha: float,
        stroke_width: float,
        stroke_color: Optional[Tuple[int, int, int]]
    ):
        """绘制竖排文字"""
        try:
            line_height = font.size
        except:
            bbox = draw.textbbox((0, 0), "测", font=font)
            line_height = bbox[3] - bbox[1]
        
        current_y = y
        for char in text:
            # 描边
            if stroke_width > 0 and stroke_color:
                for stroke_offset_x in range(-int(stroke_width), int(stroke_width) + 1):
                    for stroke_offset_y in range(-int(stroke_width), int(stroke_width) + 1):
                        if stroke_offset_x != 0 or stroke_offset_y != 0:
                            draw.text(
                                (x + stroke_offset_x, current_y + stroke_offset_y),
                                char,
                                font=font,
                                fill=(*stroke_color, int(255 * alpha))
                            )

            # 主文字
            draw.text((x, current_y), char, font=font, fill=(*color, int(255 * alpha)))
            current_y += line_height
    
    def _get_vertical_text_size(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont
    ) -> Tuple[int, int]:
        """计算竖排文字尺寸"""
        try:
            line_height = font.size
        except:
            bbox = draw.textbbox((0, 0), "测", font=font)
            line_height = bbox[3] - bbox[1]
        
        total_height = line_height * len(text)
        max_width = 0
        for char in text:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            max_width = max(max_width, char_width)
        
        return max_width, total_height
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """十六进制颜色转 RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 0)  # 默认黄色

    def render_dynamic_subtitles(
        self,
        video_path: str,
        output_path: str,
        words_list: List[Tuple[float, float, str]],
        font_path: str,
        font_size: int,
        font_color: str,
        text_direction: str,
        position: str,
        background: str,
        max_lines: int = 3,
        clearance_threshold: float = 2.0,
        stroke_width: float = 3.0,
        stroke_color: str = "black",
        subtitle_extend_time: float = 0.0,
        offset_x: int = 0,
        offset_y: int = 0
    ):
        """
        渲染动态字幕（逐词累积显示）

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            words_list: 词列表 [(start, end, word), ...]
            其他参数: 字幕样式参数
        """
        _log_info(f"🎯 开始渲染动态字幕，共 {len(words_list)} 个词")

        if not words_list:
            _log_error("❌ 逐词时间戳为空，无法渲染动态字幕")
            # 复制原视频
            import shutil
            shutil.copy(video_path, output_path)
            return

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 加载字体
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            _log_error(f"加载字体失败: {e}")
            font = ImageFont.load_default()

        # 颜色映射
        color_map = {
            "white": (255, 255, 255),
            "yellow": (255, 255, 0),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }
        font_rgb = color_map.get(font_color, (255, 255, 0))
        stroke_rgb = color_map.get(stroke_color, (0, 0, 0)) if stroke_width > 0 else None

        # 背景设置
        bg_opacity = 0.5 if background == "yes" else 0.0
        bg_rgb = (0, 0, 0)  # 黑色背景

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.video_width, self.video_height))

        # 生成动态字幕片段
        line_width_ratio = 0.9  # 固定值
        dynamic_segments = self._generate_dynamic_segments(
            words_list, max_lines, clearance_threshold, line_width_ratio, font, font_size
        )

        _log_info(f"📝 生成了 {len(dynamic_segments)} 个动态字幕片段")

        # 逐帧处理
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / self.fps

            # 查找当前时间的字幕
            current_text = None

            for start, end, text in dynamic_segments:
                # 延长字幕显示时间
                extended_end = end + subtitle_extend_time
                if start <= current_time <= extended_end:
                    current_text = text
                    break

            # 如果有字幕，绘制
            if current_text:
                # 绘制字幕（简化版，不使用动画）
                frame = self._draw_dynamic_subtitle_on_frame(
                    frame,
                    current_text,
                    font,
                    font_rgb,
                    bg_rgb,
                    bg_opacity,
                    text_direction,
                    position,
                    stroke_width,
                    stroke_rgb,
                    offset_x,
                    offset_y
                )

            out.write(frame)
            frame_idx += 1

            # 进度显示
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                _log_info(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")

        cap.release()
        out.release()

        # 使用 ffmpeg 合并音频
        _log_info(f"🎵 正在合并音频...")
        self._merge_audio(video_path, output_path)

        _log_info(f"✅ 动态字幕渲染完成: {output_path}")

    def _generate_dynamic_segments(
        self,
        words_list: List[Tuple[float, float, str]],
        max_lines: int,
        clearance_threshold: float,
        line_width_ratio: float,
        font: ImageFont.FreeTypeFont,
        font_size: int
    ) -> List[Tuple[float, float, str]]:
        """
        生成动态字幕片段（逐词累积显示）

        Returns:
            [(start, end, accumulated_text), ...]
        """
        # 检测是否是逐句时间戳（每个"词"的平均长度 > 10）
        if words_list:
            avg_word_length = sum(len(word) for _, _, word in words_list if word != "<NEWLINE>") / max(1, len([w for _, _, w in words_list if w != "<NEWLINE>"]))
            is_sentence_timestamps = avg_word_length > 10

            if is_sentence_timestamps:
                _log_info(f"🔍 检测到逐句时间戳（平均长度: {avg_word_length:.1f}），每个句子独立显示并智能换行")
        else:
            is_sentence_timestamps = False

        segments = []

        # 如果是逐句时间戳，逐句累积显示
        if is_sentence_timestamps:
            import string
            # 中文和英文标点符号
            punctuation = string.punctuation + '，。！？；：""''（）《》、·…—'

            accumulated_text = ""
            for idx, (start, end, sentence) in enumerate(words_list):
                if sentence == "<NEWLINE>":
                    continue

                # 去除标点符号
                sentence_no_punct = ''.join([c for c in sentence if c not in punctuation])

                if not sentence_no_punct.strip():
                    continue

                # 对句子进行智能换行
                max_width = int(self.video_width * 0.9)
                wrapped_sentence = smart_wrap_text(sentence_no_punct, max_width, font.path, font_size, 'zh')

                # 累积句子
                if accumulated_text:
                    accumulated_text += "\n" + wrapped_sentence
                else:
                    accumulated_text = wrapped_sentence

                segments.append((start, end, accumulated_text))
        else:
            # 逐词时间戳：累积显示
            import string
            # 中文和英文标点符号
            punctuation = string.punctuation + '，。！？；：""''（）《》、·…—'

            current_text = ""
            segment_start = 0.0
            last_word_end = 0.0

            for i, (start, end, word) in enumerate(words_list):
                # 检查是否是换行符标记（使用 <NEWLINE> 标记）
                if word == "<NEWLINE>":
                    # 在当前文本中添加换行符（保持累积，不清空）
                    if current_text:
                        current_text += "\n"
                        segments.append((segment_start, end, current_text))
                    last_word_end = end
                    continue

                # 去除标点符号
                word_no_punct = ''.join([c for c in word if c not in punctuation])

                # 如果去除标点后为空，跳过
                if not word_no_punct.strip():
                    last_word_end = end
                    continue

                # 检查是否需要清空（静音超过阈值）
                if current_text and (start - last_word_end) > clearance_threshold:
                    # 保存当前片段
                    segments.append((segment_start, last_word_end, current_text))
                    current_text = ""
                    segment_start = start

                # 累积文字
                if not current_text:
                    current_text = word_no_punct
                    segment_start = start
                else:
                    current_text += word_no_punct

                # 保存当前片段（每个词都创建一个片段）
                segments.append((segment_start, end, current_text))

                last_word_end = end

        return segments

    def _draw_dynamic_subtitle_on_frame(
        self,
        frame: np.ndarray,
        text: str,
        font: ImageFont.FreeTypeFont,
        font_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
        bg_opacity: float,
        text_direction: str,
        position: str,
        stroke_width: float,
        stroke_color: Optional[Tuple[int, int, int]],
        offset_x: int = 0,
        offset_y: int = 0
    ) -> np.ndarray:
        """在帧上绘制动态字幕（简化版）"""

        # 转换为 PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')

        # 计算文字尺寸和位置
        if text_direction == "vertical":
            text_width, text_height = self._get_vertical_text_size(draw, text, font)
        else:
            # 动态字幕：文本已经在 _generate_dynamic_segments 中进行了智能换行
            bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # 计算位置
        # 位置格式：右上、右中、右下、中上、正中、中下、左上、左中、左下
        # 兼容旧格式：bottom -> 中下, top -> 中上, middle -> 正中
        position_map = {
            "bottom": "中下",
            "top": "中上",
            "middle": "正中",
            "中中": "正中"  # 兼容旧的"中中"
        }
        position = position_map.get(position, position)

        # 解析位置
        if position in ["右上", "右中", "右下"]:
            # 右侧
            horizontal_offset = int(self.video_width * 0.05)
            x = self.video_width - text_width - horizontal_offset
        elif position in ["中上", "正中", "中下"]:
            # 中间
            x = (self.video_width - text_width) // 2
        else:  # 左上、左中、左下
            # 左侧
            horizontal_offset = int(self.video_width * 0.05)
            x = horizontal_offset

        if position in ["右上", "中上", "左上"]:
            # 上方
            vertical_offset = int(self.video_height * 0.05)
            y = vertical_offset
        elif position in ["右中", "正中", "左中"]:
            # 中间
            y = (self.video_height - text_height) // 2
        else:  # 右下、中下、左下
            # 下方
            vertical_offset = int(self.video_height * 0.05)
            y = self.video_height - text_height - vertical_offset

        # 应用偏移量
        # offset_x: 正数向右，负数向左
        # offset_y: 正数向上，负数向下（注意：图像坐标系 Y 轴向下，所以要取反）
        x += offset_x
        y -= offset_y

        # 绘制背景
        if bg_opacity > 0:
            bg_padding = 10
            bg_x1 = x - bg_padding
            bg_y1 = y - bg_padding
            bg_x2 = x + text_width + bg_padding
            bg_y2 = y + text_height + bg_padding
            draw.rectangle(
                [bg_x1, bg_y1, bg_x2, bg_y2],
                fill=(*bg_color, int(255 * bg_opacity))
            )

        # 绘制文字
        if text_direction == "vertical":
            self._draw_vertical_text(draw, text, font, x, y, font_color, 1.0, stroke_width, stroke_color)
        else:
            # 描边效果
            if stroke_width > 0 and stroke_color:
                for stroke_offset_x in range(-int(stroke_width), int(stroke_width) + 1):
                    for stroke_offset_y in range(-int(stroke_width), int(stroke_width) + 1):
                        if stroke_offset_x != 0 or stroke_offset_y != 0:
                            draw.multiline_text(
                                (x + stroke_offset_x, y + stroke_offset_y),
                                text,
                                font=font,
                                fill=(*stroke_color, 255),
                                spacing=10,
                                align='center'
                            )

            # 主文字
            draw.multiline_text(
                (x, y),
                text,
                font=font,
                fill=(*font_color, 255),
                spacing=10,
                align='center'
            )

        # 转换回 OpenCV 格式 (BGR)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _merge_audio(self, source_video: str, target_video: str):
        """
        使用 ffmpeg 将源视频的音频合并到目标视频

        Args:
            source_video: 源视频路径（包含音频）
            target_video: 目标视频路径（只有视频流，需要添加音频）
        """
        import subprocess
        import tempfile

        # 创建临时文件
        temp_output = target_video + ".temp.mp4"

        try:
            # 使用 ffmpeg 合并音频
            # -i target_video: 输入目标视频（只有视频流）
            # -i source_video: 输入源视频（包含音频）
            # -map 0:v:0: 使用第一个输入的视频流
            # -map 1:a:0?: 使用第二个输入的音频流（如果存在）
            # -c:v copy: 视频流直接复制，不重新编码
            # -c:a aac: 音频流使用 AAC 编码
            # -shortest: 以最短的流为准
            cmd = [
                'ffmpeg',
                '-i', target_video,
                '-i', source_video,
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y',  # 覆盖输出文件
                temp_output
            ]

            # 执行 ffmpeg 命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0:
                # 成功：替换原文件
                import shutil
                shutil.move(temp_output, target_video)
                _log_info(f"✅ 音频合并成功")
            else:
                # 失败：保留原文件（无音频）
                _log_error(f"❌ 音频合并失败: {result.stderr}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                _log_info(f"⚠️ 保留无音频的视频文件")

        except FileNotFoundError:
            _log_error(f"❌ 未找到 ffmpeg，无法合并音频")
            _log_error(f"⚠️ 请安装 ffmpeg: https://ffmpeg.org/download.html")
            if os.path.exists(temp_output):
                os.remove(temp_output)

        except Exception as e:
            _log_error(f"❌ 音频合并出错: {str(e)}")
            if os.path.exists(temp_output):
                os.remove(temp_output)

