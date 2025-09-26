#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断FFmpeg光流过渡失败的具体原因
"""

import os
import sys
import tempfile
import subprocess

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doubao_seed import VideoStitchingNode

def create_test_video(output_path, duration=4, color="red", width=1248, height=704, fps=24):
    """创建测试视频"""
    try:
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color={color}:size={width}x{height}:duration={duration}:rate={fps}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"创建测试视频失败: {str(e)}")
        return False

def diagnose_ffmpeg_command():
    """诊断FFmpeg命令问题"""
    print("🔍 诊断FFmpeg光流过渡失败原因...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        # 创建两个测试视频
        video1_path = os.path.join(temp_dir, "test_video_1.mp4")
        video2_path = os.path.join(temp_dir, "test_video_2.mp4")
        output_path = os.path.join(temp_dir, "output_test.mp4")
        
        print("📹 创建测试视频...")
        if not create_test_video(video1_path, duration=4, color="red"):
            print("❌ 创建视频1失败")
            return False
            
        if not create_test_video(video2_path, duration=4, color="blue"):
            print("❌ 创建视频2失败")
            return False
        
        print("✅ 测试视频创建成功")
        
        # 获取VideoStitchingNode实例
        stitching_node = VideoStitchingNode()
        
        # 分析视频属性
        video_info = stitching_node._analyze_video_properties([video1_path, video2_path])
        print(f"📊 视频信息: {video_info}")
        
        # 构建光流滤镜
        filter_complex = stitching_node._build_optical_flow_filter([video1_path, video2_path], 0.5)
        print(f"🔧 滤镜链: {filter_complex}")
        
        # 构建完整的FFmpeg命令
        quality_params = stitching_node._get_quality_params("medium")
        
        cmd = [
            'ffmpeg',
            '-i', video1_path,
            '-i', video2_path,
            '-filter_complex', filter_complex,
            '-map', '[output]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-r', str(int(video_info['target_fps'])),
            '-vsync', 'cfr',
        ] + quality_params + [
            '-y',
            output_path
        ]
        
        print(f"🚀 执行FFmpeg命令:")
        print(f"   {' '.join(cmd)}")
        
        # 执行命令并捕获详细错误
        print("\n⏳ 执行中...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"\n📊 执行结果:")
        print(f"   返回码: {result.returncode}")
        print(f"   输出文件存在: {os.path.exists(output_path)}")
        
        if result.stdout:
            print(f"\n📝 标准输出:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\n❌ 错误输出:")
            print(result.stderr)
        
        # 分析具体错误原因
        if result.returncode != 0:
            print(f"\n🔍 错误分析:")
            
            if "Invalid argument" in result.stderr:
                print("   • 可能是滤镜参数错误")
            elif "No such filter" in result.stderr:
                print("   • 可能是滤镜不存在")
            elif "Cannot determine format" in result.stderr:
                print("   • 可能是输入文件格式问题")
            elif "Permission denied" in result.stderr:
                print("   • 可能是文件权限问题")
            elif "codec" in result.stderr.lower():
                print("   • 可能是编码器问题")
            elif "filter" in result.stderr.lower():
                print("   • 可能是滤镜链语法错误")
            else:
                print("   • 未知错误，需要查看完整错误信息")
        
        # 测试简化版本
        print(f"\n🧪 测试简化版本...")
        simple_cmd = [
            'ffmpeg',
            '-i', video1_path,
            '-i', video2_path,
            '-filter_complex', '[0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5[output]',
            '-map', '[output]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            os.path.join(temp_dir, "simple_test.mp4")
        ]
        
        print(f"   简化命令: {' '.join(simple_cmd)}")
        
        simple_result = subprocess.run(
            simple_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"   简化版本返回码: {simple_result.returncode}")
        if simple_result.returncode == 0:
            print("   ✅ 简化版本成功，问题在复杂参数")
        else:
            print("   ❌ 简化版本也失败，基础环境有问题")
            if simple_result.stderr:
                print(f"   简化版本错误: {simple_result.stderr}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
        return False
    except Exception as e:
        print(f"💥 诊断过程异常: {str(e)}")
        return False

def test_ffmpeg_capabilities():
    """测试FFmpeg的功能支持"""
    print("\n🔧 测试FFmpeg功能支持...")
    
    # 测试xfade滤镜支持
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-filters'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if 'xfade' in result.stdout:
            print("✅ xfade滤镜支持")
        else:
            print("❌ xfade滤镜不支持")
            
        if 'minterpolate' in result.stdout:
            print("✅ minterpolate滤镜支持")
        else:
            print("❌ minterpolate滤镜不支持")
            
    except Exception as e:
        print(f"❌ 无法检查FFmpeg滤镜支持: {str(e)}")
    
    # 测试编码器支持
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if 'libx264' in result.stdout:
            print("✅ libx264编码器支持")
        else:
            print("❌ libx264编码器不支持")
            
    except Exception as e:
        print(f"❌ 无法检查FFmpeg编码器支持: {str(e)}")

if __name__ == "__main__":
    print("🔍 启动FFmpeg光流过渡失败诊断")
    print("="*60)
    print("🎯 本诊断将分析:")
    print("   • FFmpeg命令构建是否正确")
    print("   • 滤镜链语法是否有误")
    print("   • FFmpeg功能支持情况")
    print("   • 具体错误原因分析")
    print("="*60)
    
    # 测试FFmpeg基础功能
    test_ffmpeg_capabilities()
    
    # 诊断具体命令问题
    success = diagnose_ffmpeg_command()
    
    if success:
        print("\n🎊 诊断完成!")
        print("💡 请查看上述错误分析，找出光流过渡失败的具体原因")
    else:
        print("\n💥 诊断过程失败")
        print("💡 可能是FFmpeg环境配置问题")
