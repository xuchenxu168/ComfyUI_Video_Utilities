#!/usr/bin/env python3
"""
视频尾帧提取器 - 支持多种技术方案
"""

import os
import subprocess
import tempfile
from pathlib import Path

def extract_last_frame_ffmpeg(video_path, output_path=None):
    """
    使用FFmpeg提取视频的最后一帧
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径，如果为None则自动生成
    
    Returns:
        str: 输出图片的路径，失败返回None
    """
    try:
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return None
        
        # 如果没有指定输出路径，自动生成
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_last_frame.jpg"
        
        # FFmpeg命令：提取最后一帧
        cmd = [
            'ffmpeg',
            '-i', video_path,           # 输入视频
            '-vf', 'select=eof',        # 选择最后一帧 (end of file)
            '-vsync', 'vfr',            # 可变帧率
            '-frames:v', '1',           # 只输出1帧
            '-y',                       # 覆盖输出文件
            output_path
        ]
        
        print(f"🎬 正在提取视频尾帧: {video_path}")
        print(f"📝 FFmpeg命令: {' '.join(cmd)}")
        
        # 执行FFmpeg命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        if result.returncode == 0:
            if os.path.exists(output_path):
                print(f"✅ 尾帧提取成功: {output_path}")
                return output_path
            else:
                print(f"❌ 输出文件未生成: {output_path}")
                return None
        else:
            print(f"❌ FFmpeg执行失败:")
            print(f"   错误信息: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ FFmpeg执行超时")
        return None
    except Exception as e:
        print(f"❌ 提取尾帧失败: {str(e)}")
        return None

def extract_last_frame_ffmpeg_alternative(video_path, output_path=None):
    """
    使用FFmpeg的另一种方法提取最后一帧（通过时长计算）
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
    
    Returns:
        str: 输出图片的路径，失败返回None
    """
    try:
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return None
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_last_frame_alt.jpg"
        
        # 首先获取视频时长
        duration_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            video_path
        ]
        
        duration_result = subprocess.run(
            duration_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if duration_result.returncode != 0:
            print(f"❌ 无法获取视频时长")
            return None
        
        try:
            duration = float(duration_result.stdout.strip())
            # 提取最后0.1秒前的帧，避免可能的编码问题
            seek_time = max(0, duration - 0.1)
        except ValueError:
            print(f"❌ 无法解析视频时长")
            return None
        
        # FFmpeg命令：跳转到接近结尾的位置提取帧
        cmd = [
            'ffmpeg',
            '-ss', str(seek_time),      # 跳转到指定时间
            '-i', video_path,           # 输入视频
            '-frames:v', '1',           # 只输出1帧
            '-y',                       # 覆盖输出文件
            output_path
        ]
        
        print(f"🎬 正在提取视频尾帧 (时长方法): {video_path}")
        print(f"⏱️ 视频时长: {duration:.2f}秒，提取位置: {seek_time:.2f}秒")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"✅ 尾帧提取成功: {output_path}")
            return output_path
        else:
            print(f"❌ 尾帧提取失败: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ 提取尾帧失败: {str(e)}")
        return None

def extract_last_frame_opencv(video_path, output_path=None):
    """
    使用OpenCV提取视频的最后一帧
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
    
    Returns:
        str: 输出图片的路径，失败返回None
    """
    try:
        import cv2
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return None
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_last_frame_cv.jpg"
        
        print(f"🎬 正在使用OpenCV提取视频尾帧: {video_path}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件")
            return None
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 视频总帧数: {total_frames}")
        
        if total_frames <= 0:
            print(f"❌ 无法获取视频帧数")
            cap.release()
            return None
        
        # 跳转到最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        # 读取最后一帧
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # 保存图片
            success = cv2.imwrite(output_path, frame)
            if success:
                print(f"✅ 尾帧提取成功: {output_path}")
                return output_path
            else:
                print(f"❌ 保存图片失败")
                return None
        else:
            print(f"❌ 读取最后一帧失败")
            return None
            
    except ImportError:
        print(f"❌ OpenCV未安装，请使用: pip install opencv-python")
        return None
    except Exception as e:
        print(f"❌ OpenCV提取尾帧失败: {str(e)}")
        return None

def extract_last_frame_pillow(video_path, output_path=None):
    """
    使用Pillow (PIL) 提取视频的最后一帧
    注意：Pillow对视频支持有限，主要支持GIF等格式
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
    
    Returns:
        str: 输出图片的路径，失败返回None
    """
    try:
        from PIL import Image
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return None
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_last_frame_pil.jpg"
        
        print(f"🎬 正在使用Pillow提取视频尾帧: {video_path}")
        
        # 打开视频文件（主要支持GIF）
        with Image.open(video_path) as img:
            # 跳转到最后一帧
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                # 到达最后一帧
                img.seek(frame_count - 1)
                
                # 保存最后一帧
                img.save(output_path, 'JPEG')
                print(f"✅ 尾帧提取成功: {output_path}")
                return output_path
                
    except ImportError:
        print(f"❌ Pillow未安装，请使用: pip install Pillow")
        return None
    except Exception as e:
        print(f"❌ Pillow提取尾帧失败: {str(e)}")
        print(f"   注意：Pillow主要支持GIF格式，对MP4等格式支持有限")
        return None

def extract_last_frame_auto(video_path, output_path=None, method='ffmpeg'):
    """
    自动选择最佳方法提取视频尾帧
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
        method: 优先使用的方法 ('ffmpeg', 'opencv', 'pillow', 'auto')
    
    Returns:
        str: 输出图片的路径，失败返回None
    """
    methods = {
        'ffmpeg': extract_last_frame_ffmpeg,
        'opencv': extract_last_frame_opencv,
        'pillow': extract_last_frame_pillow
    }
    
    if method != 'auto' and method in methods:
        # 使用指定方法
        return methods[method](video_path, output_path)
    
    # 自动选择方法，按优先级尝试
    priority_methods = ['ffmpeg', 'opencv', 'pillow']
    
    for method_name in priority_methods:
        print(f"\n🔄 尝试使用 {method_name} 方法...")
        result = methods[method_name](video_path, output_path)
        if result:
            return result
        print(f"⚠️ {method_name} 方法失败，尝试下一个方法...")
    
    print(f"❌ 所有方法都失败了")
    return None

def check_dependencies():
    """检查依赖项是否可用"""
    print("🔍 检查依赖项...")
    
    # 检查FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg 可用")
        else:
            print("❌ FFmpeg 不可用")
    except:
        print("❌ FFmpeg 未安装或不在PATH中")
    
    # 检查OpenCV
    try:
        import cv2
        print(f"✅ OpenCV 可用 (版本: {cv2.__version__})")
    except ImportError:
        print("❌ OpenCV 未安装")
    
    # 检查Pillow
    try:
        from PIL import Image
        print(f"✅ Pillow 可用")
    except ImportError:
        print("❌ Pillow 未安装")

if __name__ == "__main__":
    # 检查依赖项
    check_dependencies()
    
    # 示例用法
    test_video = "test_video.mp4"  # 替换为实际的视频文件路径
    
    if os.path.exists(test_video):
        print(f"\n🎬 测试视频尾帧提取: {test_video}")
        
        # 测试不同方法
        methods = ['ffmpeg', 'opencv', 'pillow']
        
        for method in methods:
            print(f"\n{'='*50}")
            print(f"测试 {method.upper()} 方法")
            print(f"{'='*50}")
            
            result = extract_last_frame_auto(test_video, method=method)
            if result:
                print(f"🎉 {method} 方法成功: {result}")
            else:
                print(f"💔 {method} 方法失败")
    else:
        print(f"\n💡 使用示例:")
        print(f"   python video_last_frame_extractor.py")
        print(f"   # 请将 test_video.mp4 替换为实际的视频文件路径")
        
        print(f"\n🔧 函数调用示例:")
        print(f"   # FFmpeg方法（推荐）")
        print(f"   last_frame = extract_last_frame_ffmpeg('video.mp4')")
        print(f"   ")
        print(f"   # OpenCV方法")
        print(f"   last_frame = extract_last_frame_opencv('video.mp4')")
        print(f"   ")
        print(f"   # 自动选择最佳方法")
        print(f"   last_frame = extract_last_frame_auto('video.mp4')")
