#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频闪烁对比工具
用于对比改进前后的视频拼接效果
"""

import os
import sys
import subprocess
import tempfile
import time

def analyze_video_transitions(video_path, output_dir=None):
    """分析视频过渡处的帧差异"""
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return None
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    print(f"🔍 分析视频: {os.path.basename(video_path)}")
    
    try:
        # 获取视频信息
        cmd_info = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd_info, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"❌ 无法获取视频信息")
            return None
        
        import json
        info = json.loads(result.stdout)
        
        # 提取关键信息
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            print(f"❌ 未找到视频流")
            return None
        
        duration = float(video_stream.get('duration', 0))
        fps = eval(video_stream.get('r_frame_rate', '30/1'))
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        
        print(f"📊 视频信息: {width}x{height} @{fps:.2f}fps, 时长: {duration:.2f}s")
        
        # 提取关键帧进行分析
        frame_times = []
        total_frames = int(duration * fps)
        
        # 在可能的过渡点提取帧（假设每2秒有一个过渡）
        for i in range(1, int(duration // 2) + 1):
            transition_time = i * 2
            if transition_time < duration:
                # 过渡前后的帧
                frame_times.extend([
                    transition_time - 0.1,  # 过渡前
                    transition_time,        # 过渡中
                    transition_time + 0.1   # 过渡后
                ])
        
        # 提取帧
        extracted_frames = []
        for i, time_point in enumerate(frame_times):
            if time_point >= duration:
                continue
                
            frame_path = os.path.join(output_dir, f"frame_{i:03d}_{time_point:.1f}s.png")
            
            cmd_extract = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(time_point),
                '-vframes', '1',
                '-y',
                frame_path
            ]
            
            result = subprocess.run(cmd_extract, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and os.path.exists(frame_path):
                extracted_frames.append({
                    'time': time_point,
                    'path': frame_path,
                    'size': os.path.getsize(frame_path)
                })
        
        print(f"✅ 提取了 {len(extracted_frames)} 个关键帧")
        
        return {
            'video_path': video_path,
            'duration': duration,
            'fps': fps,
            'resolution': (width, height),
            'frames': extracted_frames,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None

def compare_videos(video_paths, output_dir=None):
    """对比多个视频的质量"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    print(f"🔄 开始对比 {len(video_paths)} 个视频...")
    print(f"📁 输出目录: {output_dir}")
    
    results = []
    
    for video_path in video_paths:
        if os.path.exists(video_path):
            analysis = analyze_video_transitions(video_path, 
                os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '')))
            if analysis:
                results.append(analysis)
        else:
            print(f"⚠️ 跳过不存在的文件: {video_path}")
    
    # 生成对比报告
    print("\n" + "="*60)
    print("📊 视频对比报告")
    print("="*60)
    
    for i, result in enumerate(results):
        video_name = os.path.basename(result['video_path'])
        print(f"\n🎬 视频 {i+1}: {video_name}")
        print(f"   📏 分辨率: {result['resolution'][0]}x{result['resolution'][1]}")
        print(f"   🎞️ 帧率: {result['fps']:.2f} fps")
        print(f"   ⏱️ 时长: {result['duration']:.2f} 秒")
        print(f"   🖼️ 提取帧数: {len(result['frames'])}")
        
        if result['frames']:
            avg_frame_size = sum(f['size'] for f in result['frames']) / len(result['frames'])
            print(f"   📊 平均帧大小: {avg_frame_size/1024:.1f} KB")
    
    # 生成HTML对比页面
    html_path = os.path.join(output_dir, "comparison.html")
    generate_comparison_html(results, html_path)
    
    print(f"\n🌐 生成对比页面: {html_path}")
    print("💡 用浏览器打开该页面可以直观对比视频质量")
    
    return results

def generate_comparison_html(results, html_path):
    """生成HTML对比页面"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>视频拼接质量对比</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .video-section { margin: 20px 0; border: 1px solid #ccc; padding: 15px; }
        .frame-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .frame-item { text-align: center; }
        .frame-item img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .frame-time { font-size: 12px; color: #666; }
        h1 { color: #333; }
        h2 { color: #666; }
    </style>
</head>
<body>
    <h1>🎬 视频拼接质量对比报告</h1>
    <p>生成时间: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
    
    for i, result in enumerate(results):
        video_name = os.path.basename(result['video_path'])
        html_content += f"""
    <div class="video-section">
        <h2>视频 {i+1}: {video_name}</h2>
        <p>
            📏 分辨率: {result['resolution'][0]}x{result['resolution'][1]} | 
            🎞️ 帧率: {result['fps']:.2f} fps | 
            ⏱️ 时长: {result['duration']:.2f} 秒
        </p>
        <div class="frame-grid">
"""
        
        for frame in result['frames']:
            frame_name = os.path.basename(frame['path'])
            html_content += f"""
            <div class="frame-item">
                <img src="{frame_name}" alt="Frame at {frame['time']:.1f}s">
                <div class="frame-time">{frame['time']:.1f}s ({frame['size']/1024:.1f}KB)</div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
    <div style="margin-top: 30px; padding: 15px; background-color: #f0f0f0;">
        <h3>💡 分析说明</h3>
        <ul>
            <li>每个视频提取了过渡点前后的关键帧</li>
            <li>观察帧之间的连续性，寻找突兀的变化</li>
            <li>文件大小差异可能反映压缩质量</li>
            <li>平滑的过渡应该显示渐进的变化</li>
        </ul>
    </div>
</body>
</html>
"""
    
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return True
    except Exception as e:
        print(f"❌ 生成HTML失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🎬 视频闪烁对比工具")
    print("="*50)
    
    # 查找测试生成的视频文件
    temp_dirs = []
    import glob
    
    # 在临时目录中查找测试视频
    temp_base = tempfile.gettempdir()
    pattern = os.path.join(temp_base, "tmp*", "stitched_*.mp4")
    test_videos = glob.glob(pattern)
    
    if not test_videos:
        print("❌ 未找到测试生成的视频文件")
        print("💡 请先运行 test_improved_video_stitching.py")
        return
    
    print(f"🔍 找到 {len(test_videos)} 个测试视频:")
    for video in test_videos:
        print(f"   📹 {os.path.basename(video)}")
    
    # 对比视频
    output_dir = os.path.join(os.getcwd(), f"video_comparison_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = compare_videos(test_videos, output_dir)
    
    if results:
        print(f"\n🎉 对比完成！")
        print(f"📁 结果保存在: {output_dir}")
        
        # 尝试打开HTML文件
        html_file = os.path.join(output_dir, "comparison.html")
        if os.path.exists(html_file):
            try:
                import webbrowser
                webbrowser.open(f"file://{html_file}")
                print("🌐 已在浏览器中打开对比页面")
            except:
                print(f"🌐 请手动打开: {html_file}")
    else:
        print("❌ 对比失败")

if __name__ == "__main__":
    main()
