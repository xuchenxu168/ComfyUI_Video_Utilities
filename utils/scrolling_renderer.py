"""
滚动字幕渲染器
实现专业的电影级滚动效果
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import math


def _log_info(message):
    print(f"[ScrollingRenderer] {message}")


def _log_error(message):
    print(f"[ScrollingRenderer ERROR] {message}")


class ScrollingRenderer:
    """滚动字幕渲染器"""
    
    def __init__(self, video_width: int, video_height: int, fps: float):
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
    
    def render_scrolling_text(
        self,
        video_path: str,
        output_path: str,
        text: str,
        font_path: str,
        font_size: int,
        font_color: str,
        scroll_type: str,
        scroll_speed: float,
        start_position: str,
        loop: bool,
        background_opacity: float,
        stroke_width: float,
        stroke_color: str,
        fade_in_duration: float,
        fade_out_duration: float,
        line_spacing: float,
        text_align: str,
        perspective_strength: float,
        offset_x: int = 0,
        offset_y: int = 0
    ):
        """
        渲染滚动字幕

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            text: 滚动文本
            其他参数: 字幕样式参数
        """
        _log_info(f"🎬 开始渲染滚动字幕")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / self.fps

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
            "blue": (0, 0, 255),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255)
        }
        font_rgb = color_map.get(font_color, (255, 255, 0))
        stroke_rgb = color_map.get(stroke_color, (0, 0, 0)) if stroke_width > 0 else None

        # 准备文本
        lines = text.strip().split('\n')
        
        # 计算文本总高度和最大宽度
        text_image = self._create_text_image(
            lines, font, font_rgb, stroke_width, stroke_rgb, 
            line_spacing, text_align
        )
        text_height = text_image.height
        text_width = text_image.width

        _log_info(f"📝 文本尺寸: {text_width}x{text_height}")
        _log_info(f"📹 视频尺寸: {self.video_width}x{self.video_height}")
        _log_info(f"⏱️ 视频时长: {duration:.2f}秒")
        _log_info(f"🚀 原始滚动速度: {scroll_speed} 像素/秒")

        # 计算滚动参数
        scroll_params = self._calculate_scroll_params(
            scroll_type, scroll_speed, start_position,
            text_width, text_height, duration
        )
        
        _log_info(f"🎯 滚动参数: {scroll_params}")
        
        # 计算实际需要的滚动距离和时间
        if 'start_y' in scroll_params and 'end_y' in scroll_params:
            total_distance = abs(scroll_params['start_y'] - scroll_params['end_y'])
            time_needed = total_distance / scroll_speed
            _log_info(f"📏 滚动距离: {total_distance} 像素")
            _log_info(f"⏱️ 需要时间: {time_needed:.2f} 秒 (视频时长: {duration:.2f} 秒)")
            
            if time_needed > duration:
                _log_info(f"⚠️ 警告: 滚动时间({time_needed:.2f}s) > 视频时长({duration:.2f}s)")
                _log_info(f"💡 建议: 提高滚动速度到 {total_distance / duration:.1f} 像素/秒")

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

            # 计算当前帧的透明度（淡入淡出）
            alpha = self._calculate_fade_alpha(
                current_time, duration, fade_in_duration, fade_out_duration
            )

            if alpha > 0:
                # 根据滚动类型绘制字幕
                frame = self._draw_scrolling_text(
                    frame, text_image, scroll_type, scroll_params,
                    current_time, alpha, background_opacity,
                    perspective_strength, loop, duration, offset_x, offset_y
                )

            out.write(frame)
            frame_idx += 1

            # 进度显示（每30帧显示一次，约1秒）
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                _log_info(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}) - 时间: {current_time:.2f}s, Alpha: {alpha:.2f}")

        cap.release()
        out.release()

        # 使用 ffmpeg 合并音频
        _log_info(f"🎵 正在合并音频...")
        self._merge_audio(video_path, output_path)

        _log_info(f"✅ 滚动字幕渲染完成: {output_path}")

    def _create_text_image(
        self,
        lines: list,
        font: ImageFont.FreeTypeFont,
        font_color: Tuple[int, int, int],
        stroke_width: float,
        stroke_color: Tuple[int, int, int],
        line_spacing: float,
        text_align: str
    ) -> Image.Image:
        """创建文本图像"""
        
        # 计算每行的尺寸
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        line_heights = []
        line_widths = []
        
        for line in lines:
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        # 计算总尺寸
        max_width = max(line_widths) if line_widths else 100
        line_height = max(line_heights) if line_heights else font.size
        total_height = int(line_height * line_spacing * len(lines))
        
        # 添加边距
        padding = int(stroke_width * 2 + 20)
        img_width = max_width + padding * 2
        img_height = total_height + padding * 2
        
        # 创建透明图像
        text_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # 绘制每一行
        y = padding
        for i, line in enumerate(lines):
            line_width = line_widths[i]
            
            # 计算 x 位置（对齐）
            if text_align == "center":
                x = (img_width - line_width) // 2
            elif text_align == "right":
                x = img_width - line_width - padding
            else:  # left
                x = padding
            
            # 绘制描边
            if stroke_width > 0 and stroke_color:
                for offset_x in range(-int(stroke_width), int(stroke_width) + 1):
                    for offset_y in range(-int(stroke_width), int(stroke_width) + 1):
                        if offset_x != 0 or offset_y != 0:
                            draw.text(
                                (x + offset_x, y + offset_y),
                                line,
                                font=font,
                                fill=(*stroke_color, 255)
                            )
            
            # 绘制主文字
            draw.text((x, y), line, font=font, fill=(*font_color, 255))
            
            y += int(line_height * line_spacing)
        
        return text_img

    def _calculate_scroll_params(
        self,
        scroll_type: str,
        scroll_speed: float,
        start_position: str,
        text_width: int,
        text_height: int,
        duration: float
    ) -> dict:
        """计算滚动参数"""
        
        params = {
            'scroll_speed': scroll_speed,
            'text_width': text_width,
            'text_height': text_height
        }
        
        # 根据滚动类型计算起始位置
        if scroll_type == "vertical_up":
            # 垂直向上滚动
            if start_position == "off_screen":
                params['start_y'] = self.video_height  # 文本顶部在屏幕底部下方
            elif start_position == "bottom":
                params['start_y'] = self.video_height - 100  # 文本底部对齐屏幕底部，留100像素可见
            elif start_position == "center":
                params['start_y'] = (self.video_height - text_height) // 2
            else:  # top
                params['start_y'] = 0
            
            params['end_y'] = -text_height
            params['x'] = (self.video_width - text_width) // 2
            
        elif scroll_type == "vertical_down":
            # 垂直向下滚动
            if start_position == "off_screen":
                params['start_y'] = -text_height
            elif start_position == "top":
                params['start_y'] = 0
            elif start_position == "center":
                params['start_y'] = (self.video_height - text_height) // 2
            else:  # bottom
                params['start_y'] = self.video_height - text_height
            
            params['end_y'] = self.video_height
            params['x'] = (self.video_width - text_width) // 2
            
        elif scroll_type.startswith("horizontal_left"):
            # 水平向左滚动
            if start_position == "off_screen":
                params['start_x'] = self.video_width
            else:
                params['start_x'] = self.video_width - text_width
            
            params['end_x'] = -text_width
            
            # 根据滚动类型确定Y位置
            if scroll_type == "horizontal_left_top":
                params['y'] = int(self.video_height * 0.1)  # 顶部10%位置
            elif scroll_type == "horizontal_left_bottom":
                params['y'] = int(self.video_height * 0.9 - text_height)  # 底部10%位置
            else:  # horizontal_left_center 或 horizontal_left（兼容旧版）
                params['y'] = (self.video_height - text_height) // 2  # 中部
            
        elif scroll_type.startswith("horizontal_right"):
            # 水平向右滚动
            if start_position == "off_screen":
                params['start_x'] = -text_width
            else:
                params['start_x'] = 0
            
            params['end_x'] = self.video_width
            
            # 根据滚动类型确定Y位置
            if scroll_type == "horizontal_right_top":
                params['y'] = int(self.video_height * 0.1)  # 顶部10%位置
            elif scroll_type == "horizontal_right_bottom":
                params['y'] = int(self.video_height * 0.9 - text_height)  # 底部10%位置
            else:  # horizontal_right_center 或 horizontal_right（兼容旧版）
                params['y'] = (self.video_height - text_height) // 2  # 中部
            
        elif scroll_type == "star_wars":
            # 星战式3D透视滚动
            if start_position == "off_screen":
                params['start_y'] = self.video_height
            elif start_position == "bottom":
                params['start_y'] = self.video_height - 100  # 留100像素可见
            elif start_position == "center":
                params['start_y'] = (self.video_height - text_height) // 2
            else:  # top
                params['start_y'] = 0
            
            params['end_y'] = -text_height
            params['x'] = (self.video_width - text_width) // 2
            
        elif scroll_type == "fade_scroll":
            # 渐变滚动
            if start_position == "off_screen":
                params['start_y'] = self.video_height
            elif start_position == "bottom":
                params['start_y'] = self.video_height - 100  # 留100像素可见
            elif start_position == "center":
                params['start_y'] = (self.video_height - text_height) // 2
            else:  # top
                params['start_y'] = 0
            
            params['end_y'] = -text_height
            params['x'] = (self.video_width - text_width) // 2
        
        return params

    def _calculate_fade_alpha(
        self,
        current_time: float,
        duration: float,
        fade_in_duration: float,
        fade_out_duration: float
    ) -> float:
        """计算淡入淡出透明度"""
        
        alpha = 1.0
        
        # 淡入
        if current_time < fade_in_duration:
            alpha = current_time / fade_in_duration if fade_in_duration > 0 else 1.0
        
        # 淡出
        time_until_end = duration - current_time
        if time_until_end < fade_out_duration:
            alpha = min(alpha, time_until_end / fade_out_duration if fade_out_duration > 0 else 1.0)
        
        return max(0.0, min(1.0, alpha))

    def _draw_scrolling_text(
        self,
        frame: np.ndarray,
        text_img: Image.Image,
        scroll_type: str,
        params: dict,
        current_time: float,
        alpha: float,
        background_opacity: float,
        perspective_strength: float,
        loop: bool,
        duration: float,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> np.ndarray:
        """在帧上绘制滚动文本"""
        
        # 转换为 PIL Image
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 计算当前位置
        if scroll_type in ["vertical_up", "vertical_down", "fade_scroll"]:
            # 垂直滚动
            distance = abs(params['start_y'] - params['end_y'])
            if distance > 0:
                progress = (current_time * params['scroll_speed']) / distance

                # 循环逻辑
                if loop:
                    # 使用模运算实现循环
                    progress = progress % 1.0
                else:
                    # 不循环，限制在0-1之间
                    progress = min(progress, 1.0)
            else:
                progress = 0.0

            if scroll_type == "vertical_up" or scroll_type == "fade_scroll" or scroll_type == "star_wars":
                # 向上滚动
                current_y = int(params['start_y'] - progress * distance)
            else:
                # 向下滚动
                current_y = int(params['start_y'] + progress * distance)

            current_x = params['x']

            # 应用偏移量
            # offset_x: 正数向右，负数向左
            # offset_y: 正数向上，负数向下（注意：图像坐标系 Y 轴向下，所以要取反）
            current_x += offset_x
            current_y -= offset_y

            if scroll_type == "fade_scroll":
                # 渐变滚动：根据位置调整透明度
                screen_progress = (params['start_y'] - current_y) / (params['start_y'] + params['text_height'])
                alpha *= self._calculate_gradient_alpha(screen_progress)
            
        elif scroll_type.startswith("horizontal_left") or scroll_type.startswith("horizontal_right"):
            # 水平滚动
            distance = abs(params['start_x'] - params['end_x'])
            if distance > 0:
                progress = (current_time * params['scroll_speed']) / distance

                # 循环逻辑
                if loop:
                    # 使用模运算实现循环
                    progress = progress % 1.0
                else:
                    # 不循环，限制在0-1之间
                    progress = min(progress, 1.0)
            else:
                progress = 0.0

            if scroll_type.startswith("horizontal_left"):
                current_x = int(params['start_x'] - progress * distance)
            else:
                current_x = int(params['start_x'] + progress * distance)

            current_y = params['y']

            # 应用偏移量
            # offset_x: 正数向右，负数向左
            # offset_y: 正数向上，负数向下（注意：图像坐标系 Y 轴向下，所以要取反）
            current_x += offset_x
            current_y -= offset_y
            
        elif scroll_type == "star_wars":
            # 星战式3D透视滚动
            distance = abs(params['start_y'] - params['end_y'])
            if distance > 0:
                progress = (current_time * params['scroll_speed']) / distance

                # 循环逻辑
                if loop:
                    # 使用模运算实现循环
                    progress = progress % 1.0
                else:
                    # 不循环，限制在0-1之间
                    progress = min(progress, 1.0)
            else:
                progress = 0.0

            current_y = int(params['start_y'] - progress * distance)

            # 应用3D透视变换
            text_img = self._apply_perspective_transform(
                text_img, current_y, perspective_strength
            )

            current_x = (self.video_width - text_img.width) // 2

            # 应用偏移量
            # offset_x: 正数向右，负数向左
            # offset_y: 正数向上，负数向下（注意：图像坐标系 Y 轴向下，所以要取反）
            current_x += offset_x
            current_y -= offset_y
        
        # 应用透明度
        if alpha < 1.0:
            text_img = self._apply_alpha(text_img, alpha)
        
        # 绘制背景（如果需要）
        if background_opacity > 0:
            bg_layer = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_layer)
            
            bg_padding = 20
            bg_x1 = max(0, current_x - bg_padding)
            bg_y1 = max(0, current_y - bg_padding)
            bg_x2 = min(self.video_width, current_x + text_img.width + bg_padding)
            bg_y2 = min(self.video_height, current_y + text_img.height + bg_padding)
            
            bg_draw.rectangle(
                [bg_x1, bg_y1, bg_x2, bg_y2],
                fill=(0, 0, 0, int(255 * background_opacity * alpha))
            )
            
            pil_frame = Image.alpha_composite(pil_frame.convert('RGBA'), bg_layer)
        
        # 粘贴文本（确保转换为RGBA模式）
        if pil_frame.mode != 'RGBA':
            pil_frame = pil_frame.convert('RGBA')
        
        # 调试：每秒输出一次位置信息
        if int(current_time * 10) % 10 == 0:  # 每0.1秒
            _log_info(f"📍 位置: x={current_x}, y={current_y}, 文本尺寸: {text_img.width}x{text_img.height}, 屏幕: {self.video_width}x{self.video_height}")
        
        # 粘贴文本（只要有部分在屏幕内就粘贴）
        if (current_x + text_img.width > 0 and current_x < self.video_width and
            current_y + text_img.height > 0 and current_y < self.video_height):
            pil_frame.paste(text_img, (current_x, current_y), text_img)
        else:
            # 调试：文本不在可见区域
            if int(current_time * 10) % 10 == 0:
                _log_info(f"⚠️ 文本不在可见区域")
        
        # 转换回 OpenCV 格式
        return cv2.cvtColor(np.array(pil_frame.convert('RGB')), cv2.COLOR_RGB2BGR)

    def _apply_perspective_transform(
        self,
        img: Image.Image,
        y_position: int,
        strength: float
    ) -> Image.Image:
        """应用3D透视变换（星战效果）"""
        
        if strength <= 0:
            return img
        
        # 计算透视比例
        screen_ratio = 1.0 - (y_position / self.video_height)
        screen_ratio = max(0.1, min(1.0, screen_ratio))
        
        # 应用透视缩放
        scale_factor = 0.3 + 0.7 * screen_ratio * (1.0 - strength * 0.7)
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        if new_width > 0 and new_height > 0:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img

    def _calculate_gradient_alpha(self, progress: float) -> float:
        """计算渐变透明度"""
        
        # 在屏幕顶部和底部渐变
        if progress < 0.2:
            return progress / 0.2
        elif progress > 0.8:
            return (1.0 - progress) / 0.2
        else:
            return 1.0

    def _apply_alpha(self, img: Image.Image, alpha: float) -> Image.Image:
        """应用透明度"""
        
        img_with_alpha = img.copy()
        
        # 获取 alpha 通道
        if img_with_alpha.mode != 'RGBA':
            img_with_alpha = img_with_alpha.convert('RGBA')
        
        # 调整 alpha 通道
        alpha_channel = img_with_alpha.split()[3]
        alpha_channel = alpha_channel.point(lambda p: int(p * alpha))
        img_with_alpha.putalpha(alpha_channel)
        
        return img_with_alpha

    def _merge_audio(self, source_video: str, target_video: str):
        """使用 ffmpeg 将源视频的音频合并到目标视频"""
        import subprocess
        
        temp_output = target_video + ".temp.mp4"
        
        try:
            cmd = [
                'ffmpeg',
                '-i', target_video,
                '-i', source_video,
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y',
                temp_output
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                import shutil
                shutil.move(temp_output, target_video)
                _log_info(f"✅ 音频合并成功")
            else:
                _log_error(f"❌ 音频合并失败: {result.stderr}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
        
        except FileNotFoundError:
            _log_error(f"❌ 未找到 ffmpeg，无法合并音频")
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        except Exception as e:
            _log_error(f"❌ 音频合并出错: {str(e)}")
            if os.path.exists(temp_output):
                os.remove(temp_output)