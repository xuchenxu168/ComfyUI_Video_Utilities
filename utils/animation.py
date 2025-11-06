"""
动画效果模块
保留所有 11 种动画特效
"""

import math
from typing import Tuple


def apply_animation(
    animation: str,
    progress: float,
    subtitle_progress: float,
    base_x: int,
    base_y: int,
    base_color: Tuple[int, int, int],
    width: int,
    height: int,
    text_height: int
) -> Tuple[int, int, float, float, Tuple[int, int, int]]:
    """
    应用字幕动画效果
    
    Args:
        animation: 动画类型
        progress: 动画进度 (0.0 到 1.0)
        subtitle_progress: 字幕整体进度 (0.0 到 1.0)
        base_x, base_y: 基础位置
        base_color: 基础颜色 (R, G, B)
        width, height: 视频尺寸
        text_height: 文字高度
    
    Returns:
        (x, y, alpha, scale, color): 调整后的位置、透明度、缩放、颜色
    """
    x, y = base_x, base_y
    alpha = 1.0
    scale = 1.0
    color = base_color
    
    if animation == "none":
        pass
    
    elif animation == "fade_in":
        # 淡入效果
        alpha = progress
    
    elif animation == "slide_up":
        # 从下滑入
        offset = int(text_height * 2 * (1 - progress))
        y = base_y + offset
        alpha = progress
    
    elif animation == "slide_down":
        # 从上滑入
        offset = int(text_height * 2 * (1 - progress))
        y = base_y - offset
        alpha = progress
    
    elif animation == "zoom_in":
        # 放大效果
        scale = 0.5 + 0.5 * progress
        alpha = progress
    
    elif animation == "typewriter":
        # 打字机效果（通过透明度模拟）
        alpha = min(progress * 3, 1.0)
    
    elif animation == "bounce":
        # 弹跳效果
        if progress < 0.5:
            bounce = math.sin(progress * math.pi * 4) * 20 * (1 - progress * 2)
        else:
            bounce = 0
        y = base_y - int(bounce)
        alpha = min(progress * 2, 1.0)
    
    elif animation == "wave":
        # 波浪效果
        wave = math.sin(subtitle_progress * math.pi * 4) * 10
        y = base_y + int(wave)
    
    elif animation == "glow":
        # 发光效果（通过缩放模拟）
        glow = 1.0 + math.sin(subtitle_progress * math.pi * 6) * 0.05
        scale = glow
    
    elif animation == "rainbow":
        # 彩虹色效果
        hue = (subtitle_progress * 360) % 360
        # HSV 转 RGB
        h = hue / 60
        c = 255
        x_val = int(c * (1 - abs(h % 2 - 1)))
        
        if 0 <= h < 1:
            color = (c, x_val, 0)
        elif 1 <= h < 2:
            color = (x_val, c, 0)
        elif 2 <= h < 3:
            color = (0, c, x_val)
        elif 3 <= h < 4:
            color = (0, x_val, c)
        elif 4 <= h < 5:
            color = (x_val, 0, c)
        else:
            color = (c, 0, x_val)
    
    elif animation == "karaoke":
        # 卡拉OK效果（颜色渐变）
        if subtitle_progress < 0.5:
            # 前半段：白色到黄色
            t = subtitle_progress * 2
            color = (
                int(255 * (1 - t) + 255 * t),    # R
                int(255 * (1 - t) + 255 * t),    # G
                int(255 * (1 - t) + 0 * t)       # B
            )
        else:
            # 后半段：黄色到红色
            t = (subtitle_progress - 0.5) * 2
            color = (
                int(255 * (1 - t) + 255 * t),    # R
                int(255 * (1 - t) + 0 * t),      # G
                int(0 * (1 - t) + 0 * t)         # B
            )
    
    elif animation == "rotate_in":
        # 旋转进入效果（通过缩放和透明度模拟）
        if progress < 1.0:
            scale = 0.3 + 0.7 * progress
            alpha = progress
            # 添加轻微的位置偏移来模拟旋转
            offset = int(20 * (1 - progress) * math.sin(progress * math.pi * 2))
            x = base_x + offset
        # else: 使用默认值 (scale=1.0, alpha=1.0)
    
    elif animation == "flip_in":
        # 翻转进入效果（通过缩放模拟3D翻转）
        if progress < 0.5:
            # 前半段：缩小到0
            scale = max(0.3, 1.0 - progress * 2)
            alpha = max(0.3, 1.0 - progress * 2)
        elif progress < 1.0:
            # 后半段：从0放大
            scale = (progress - 0.5) * 2
            alpha = (progress - 0.5) * 2
        # else: 使用默认值 (scale=1.0, alpha=1.0)
    
    elif animation == "elastic":
        # 弹性效果（类似橡皮筋）
        if progress < 1.0:
            # 使用弹性缓动函数
            p = 0.3
            s = p / 4
            t = progress
            # 计算弹性缩放，确保不会小于0.3
            elastic_scale = math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1
            scale = max(0.3, min(1.5, elastic_scale))  # 限制在0.3-1.5之间
            alpha = min(progress * 2, 1.0)
        # else: 使用默认值 (scale=1.0, alpha=1.0)
    
    elif animation == "shake":
        # 抖动效果
        if progress < 0.5:
            # 入场时抖动
            shake_x = int(math.sin(progress * math.pi * 20) * 10 * (1 - progress * 2))
            x = base_x + shake_x
            alpha = progress * 2
        # else: 使用默认值 (alpha=1.0)
    
    elif animation == "pulse":
        # 脉冲效果（持续的缩放脉动）
        pulse = 1.0 + math.sin(subtitle_progress * math.pi * 8) * 0.1
        scale = pulse
        # alpha 保持默认值 1.0
    
    elif animation == "swing":
        # 摆动效果（像钟摆一样）
        if progress < 0.8:
            # 摆动幅度逐渐减小
            swing_angle = math.sin(progress * math.pi * 6) * 30 * (1 - progress)
            offset_x = int(swing_angle)
            x = base_x + offset_x
            alpha = min(progress * 1.5, 1.0)
        # else: 使用默认值 (alpha=1.0)
    
    elif animation == "blur_in":
        # 模糊进入效果（通过快速缩放模拟）
        if progress < 0.5:
            scale = 1.5 - progress
            alpha = progress * 2
        elif progress < 1.0:
            scale = 1.0
            alpha = 1.0
        # else: 使用默认值 (scale=1.0, alpha=1.0)
    
    elif animation == "neon":
        # 霓虹灯效果（颜色闪烁 + 发光）
        # 颜色在青色和品红色之间变化
        t = math.sin(subtitle_progress * math.pi * 10) * 0.5 + 0.5
        color = (
            int(255 * t),           # R
            int(100 + 155 * (1-t)), # G
            int(255 * (1-t))        # B
        )
        # 添加脉冲效果
        scale = 1.0 + math.sin(subtitle_progress * math.pi * 10) * 0.08
        # alpha 保持默认值 1.0
    
    return (x, y, alpha, scale, color)


def get_animation_list():
    """获取所有支持的动画效果列表"""
    return [
        "none",
        "fade_in",
        "slide_up",
        "slide_down",
        "zoom_in",
        "typewriter",
        "bounce",
        "wave",
        "glow",
        "rainbow",
        "karaoke",
        "rotate_in",
        "flip_in",
        "elastic",
        "shake",
        "pulse",
        "swing",
        "blur_in",
        "neon"
    ]

