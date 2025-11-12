"""
Doubao-Seed 节点
基于ComfyUI_Comfly项目的Doubao Seedream4实现
集成多家API调用，支持图像生成和视频生成功能
"""

import os
import json
import requests
import time
import random
import base64
import io
import subprocess
from PIL import Image
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import urllib3
import ssl
import glob
import torchaudio

# 导入 ComfyUI 的文件夹路径
try:
    import folder_paths
    input_dir = folder_paths.get_input_directory()
    output_dir = folder_paths.get_output_directory()
except ImportError:
    # 如果无法导入 folder_paths，使用默认路径
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "input")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile
import shutil
from urllib.parse import urlparse
from fractions import Fraction

# 导入ComfyUI的视频类型 - 使用官方标准
try:
    from comfy_api.input_impl import VideoFromFile
    HAS_COMFYUI_VIDEO = True
    print("[Video Utilities] 信息：✅ ComfyUI官方视频类型导入成功")
except ImportError as e:
    HAS_COMFYUI_VIDEO = False
    print(f"[Video Utilities] 信息：⚠️ ComfyUI视频类型导入失败，使用简化版本: {e}")
    # 创建简单的替代类
    class VideoFromFile:
        def __init__(self, file_path):
            self.file_path = file_path
        def get_dimensions(self):
            return (512, 512)  # 默认尺寸

# 尝试导入视频处理库
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    _log_warning("OpenCV未安装，视频处理功能受限")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_ssl_compatible_session():
    """创建SSL兼容的requests session"""
    session = requests.Session()

    # 配置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # 创建自定义的HTTPAdapter
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            # 创建更宽松的SSL上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # 支持更多SSL协议版本和密码套件
            try:
                ssl_context.minimum_version = ssl.TLSVersion.TLSv1
                ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
            except AttributeError:
                # 兼容旧版本Python
                pass

            # 设置更宽松的密码套件
            try:
                ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')
            except ssl.SSLError:
                try:
                    ssl_context.set_ciphers('ALL:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')
                except ssl.SSLError:
                    pass  # 使用默认密码套件

            # 禁用各种SSL检查
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            kwargs['ssl_context'] = ssl_context
            return super().init_poolmanager(*args, **kwargs)

    # 应用适配器
    adapter = SSLAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # 设置超时和其他选项
    session.verify = False  # 禁用SSL验证

    return session

# 全局常量和配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDREAM4_CONFIG_FILE = 'SeedReam4_config.json'

def _log_info(message):
    print(f"[Video Utilities] 信息：{message}")

def _log_warning(message):
    print(f"[Video Utilities] 警告：{message}")

def _log_error(message):
    print(f"[Video Utilities] 错误：{message}")
def tensor2pil(tensor):
    """将tensor转换为PIL图像 - 支持ComfyUI的[B, H, W, C]格式"""
    if tensor is None:
        _log_warning("⚠️ tensor2pil: 输入tensor为None")
        return None
    if isinstance(tensor, list):
        return [tensor2pil(img) for img in tensor]

    try:
        # 确保tensor是4维的
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        # 如果是batch，处理多图情况（这里只处理单图转换，多图拼接在image_to_base64中处理）
        if len(tensor.shape) == 4 and tensor.shape[0] > 1:
            # 对于tensor2pil函数，我们只转换第一张图像
            # 多图拼接逻辑在image_to_base64函数中处理
            tensor = tensor[0:1]

        # 现在应该是 [1, H, W, C] 格式，去掉batch维度
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # 变成 [H, W, C]

        # 检查是否需要转换通道顺序
        if len(tensor.shape) == 3:
            # 如果最后一个维度不是3（RGB通道），可能是[C, H, W]格式
            if tensor.shape[-1] != 3 and tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

        # 转换为numpy数组
        if hasattr(tensor, 'cpu'):
            # PyTorch tensor
            np_image = tensor.cpu().numpy()
        else:
            # 已经是numpy数组
            np_image = tensor

        # 确保数据类型和范围正确
        if np_image.dtype != np.uint8:
            if np_image.max() <= 1.0:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np.clip(np_image, 0, 255).astype(np.uint8)

        # 如果是灰度图像，转换为RGB
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[-1] == 1:
            np_image = np.repeat(np_image, 3, axis=-1)

        pil_image = Image.fromarray(np_image)

        return pil_image

    except Exception as e:
        _log_error(f"❌ tensor2pil转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def pil2tensor(image):
    """将PIL图像转换为tensor - 参考ComfyUI_Comfly的实现"""
    if image is None:
        return None
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    
    # 转换为RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为numpy数组并归一化到[0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 返回tensor，格式为[1, H, W, 3] - 这是ComfyUI的标准格式
    return torch.from_numpy(img_array)[None,]

def create_blank_tensor(width=1024, height=1024):
    """创建正确格式的空白tensor - 参考ComfyUI_Comfly的实现"""
    blank_image = Image.new('RGB', (width, height), color='white')
    np_image = np.array(blank_image).astype(np.float32) / 255.0
    # 返回tensor，格式为[1, H, W, 3] - 这是ComfyUI的标准格式
    return torch.from_numpy(np_image)[None,]

def ensure_tensor_format(tensor):
    """确保tensor格式完全符合ComfyUI要求 - 格式为[1, H, W, 3]"""
    if tensor is None:
        return create_blank_tensor()
    
    original_shape = tensor.shape
    _log_info(f"🔍 输入tensor形状: {original_shape}")
    
    # 处理特殊情况：如果tensor形状是 (1, 1, 2048) 或类似格式
    if len(tensor.shape) == 3 and tensor.shape[1] == 1 and tensor.shape[2] > 1000:
        _log_warning(f"⚠️ 检测到异常tensor形状: {tensor.shape}，可能是1D数据被错误reshape")
        return create_blank_tensor()
    
    # 确保是4维tensor，格式为[1, H, W, 3]
    if len(tensor.shape) != 4:
        if len(tensor.shape) == 3:
            # 检查是否是 (H, W, 3) 格式
            if tensor.shape[-1] == 3:
                tensor = tensor.unsqueeze(0)
                _log_info(f"🔧 添加batch维度: {tensor.shape}")
            else:
                _log_error(f"❌ 无法修复tensor维度: {original_shape}")
                return create_blank_tensor()
        else:
            _log_error(f"❌ 无法修复tensor维度: {original_shape}")
            return create_blank_tensor()
    
    # 确保是 (batch, height, width, channels) 格式
    if tensor.shape[-1] != 3:
        if tensor.shape[1] == 3:  # 如果是 (batch, channels, height, width) 格式
            tensor = tensor.permute(0, 2, 3, 1)
            _log_info(f"🔧 重新排列tensor维度: {tensor.shape}")
        else:
            _log_error(f"❌ 无法修复tensor通道维度: {tensor.shape}")
            return create_blank_tensor()
    
    # 确保数据类型正确
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
        _log_info(f"🔧 转换tensor数据类型: {tensor.dtype}")
    
    # 确保值范围正确 (0-1)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = torch.clamp(tensor, 0, 1)
        _log_info(f"🔧 限制tensor值范围: {tensor.min().item():.3f} 到 {tensor.max().item():.3f}")
    
    # 确保没有异常值
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        _log_error("❌ tensor包含异常值，使用空白tensor替代")
        return create_blank_tensor()
    
    # 最终验证 - 确保是[1, H, W, 3]格式
    if len(tensor.shape) != 4 or tensor.shape[-1] != 3:
        _log_error(f"❌ 最终tensor格式仍然不正确: {tensor.shape}")
        return create_blank_tensor()
    
    _log_info(f"✅ tensor格式验证通过: {tensor.shape}")
    return tensor

def image_to_base64(image_tensor, max_size=2048, return_data_url=True):
    """将tensor转换为base64字符串，支持自动压缩和多图拼接

    Args:
        image_tensor: 输入的图像tensor
        max_size: 最大尺寸限制
        return_data_url: 是否返回完整的data URL格式，False则只返回base64字符串
    """
    if image_tensor is None:
        return None

    # 如果是batch，将多张图像水平拼接成一张大图
    if len(image_tensor.shape) == 4 and image_tensor.shape[0] > 1:
        _log_info(f"🔍 检测到多图batch输入 {image_tensor.shape}，将拼接成一张大图")

        # 将每张图像转换为PIL图像
        pil_images = []
        for i in range(image_tensor.shape[0]):
            single_tensor = image_tensor[i:i+1]  # 保持4D格式
            pil_img = tensor2pil(single_tensor)
            if pil_img is not None:
                pil_images.append(pil_img)

        if not pil_images:
            return None

        # 水平拼接图像
        total_width = sum(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)

        # 创建拼接后的大图
        combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img in pil_images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        pil_image = combined_image
        _log_info(f"🔧 多图拼接完成: {len(pil_images)}张图 -> {pil_image.size}")
    else:
        pil_image = tensor2pil(image_tensor)
        if pil_image is None:
            return None

    # 检查图像尺寸，如果过大则压缩
    original_size = pil_image.size
    if max(original_size) > max_size:
        # 计算新尺寸，保持宽高比
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        _log_info(f"🔧 图像压缩: {original_size} -> {new_size}")

    buffered = io.BytesIO()
    # 使用JPEG格式压缩大图像，PNG格式保留小图像
    if max(pil_image.size) > 1024:
        # 对于图像编辑，使用更高质量的JPEG
        quality = 90 if max(original_size) > max_size else 85
        pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
        format_prefix = "data:image/jpeg;base64,"
    else:
        pil_image.save(buffered, format="PNG", optimize=True)
        format_prefix = "data:image/png;base64,"

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 验证base64字符串的有效性
    try:
        # 尝试解码验证
        base64.b64decode(image_base64)
    except Exception as e:
        _log_warning(f"⚠️ Base64编码验证失败: {e}")
        return None

    if return_data_url:
        return f"{format_prefix}{image_base64}"
    else:
        return image_base64

def download_video_from_url(video_url: str, output_dir: str = None) -> str:
    """从URL下载视频文件"""
    try:
        if not video_url or not video_url.strip():
            raise ValueError("视频URL为空")

        # 解析URL获取文件名
        parsed_url = urlparse(video_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = f"video_{int(time.time())}.mp4"

        # 确定输出目录
        if output_dir is None:
            # 使用ComfyUI的输出目录
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except:
                output_dir = tempfile.gettempdir()

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 完整的输出路径
        output_path = os.path.join(output_dir, filename)

        _log_info(f"🔽 开始下载视频: {video_url}")
        _log_info(f"📁 保存路径: {output_path}")

        # 下载视频
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()

        # 写入文件
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(output_path)
        _log_info(f"✅ 视频下载完成: {filename} ({file_size / 1024 / 1024:.2f} MB)")

        return output_path

    except Exception as e:
        _log_error(f"视频下载失败: {e}")
        return None

def create_preview_compatible_video(video_path: str, force_convert=False, use_cache=True):
    """为 MPEG-4 等不兼容格式创建预览兼容的 H.264 版本

    Args:
        video_path: 原始视频路径
        force_convert: 是否强制重新转换
        use_cache: 是否使用缓存的预览版本

    Returns:
        str: 预览版本的路径，如果转换失败则返回 None
    """
    try:
        if not video_path or not os.path.exists(video_path):
            _log_error(f"❌ 视频文件不存在: {video_path}")
            return None

        # 确定预览文件的保存位置
        # 如果原视频在 input 目录，预览版本也放在 input 目录
        # 如果原视频在 output 目录，预览版本也放在 output 目录
        video_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        preview_filename = f"{base_name}_h264_preview.mp4"
        preview_path = os.path.join(video_dir, preview_filename)

        # 检查是否已经存在预览版本
        if use_cache and os.path.exists(preview_path) and not force_convert:
            # 检查预览版本是否比原视频新
            if os.path.getmtime(preview_path) >= os.path.getmtime(video_path):
                _log_info(f"🎬 使用缓存的预览版本: {preview_path}")
                return preview_path
            else:
                _log_info(f"🔄 预览版本过期，需要重新创建")

        # 检查原视频编码
        import subprocess
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

        if probe_result.returncode != 0:
            _log_error(f"❌ 无法检测视频编码: {video_path}")
            return None

        codec_name = probe_result.stdout.strip()
        _log_info(f"🎬 检测到视频编码: {codec_name}")

        # 只为 MPEG-4 等问题编码创建预览版本
        if codec_name not in ['mpeg4', 'xvid', 'divx']:
            _log_info(f"✅ 视频编码 {codec_name} 通常被浏览器支持，无需转换")
            return None

        _log_info(f"🔧 为 {codec_name} 编码创建 H.264 预览版本...")
        _log_info(f"📁 预览文件将保存到: {preview_path}")

        # 创建 H.264 预览版本
        # 使用快速编码参数以减少转换时间
        convert_cmd = [
            "ffmpeg", "-i", video_path,
            "-c:v", "libx264",           # H.264 编码
            "-preset", "veryfast",       # 快速编码
            "-crf", "23",                # 质量参数（23 是较好的质量）
            "-c:a", "aac",               # AAC 音频编码
            "-b:a", "128k",              # 音频比特率
            "-movflags", "+faststart",   # 优化网络播放
            "-y",                        # 覆盖已存在的文件
            preview_path
        ]

        _log_info(f"🎬 开始转换...")
        _log_info(f"命令: {' '.join(convert_cmd)}")

        # 执行转换
        convert_result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if convert_result.returncode == 0 and os.path.exists(preview_path):
            file_size = os.path.getsize(preview_path) / (1024 * 1024)
            _log_info(f"✅ 预览版本创建成功: {preview_path} ({file_size:.2f} MB)")
            return preview_path
        else:
            _log_error(f"❌ 预览版本创建失败")
            _log_error(f"FFmpeg 错误: {convert_result.stderr}")
            # 清理失败的文件
            if os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                except:
                    pass
            return None

    except subprocess.TimeoutExpired:
        _log_error(f"❌ 视频转换超时（超过5分钟）")
        return None
    except Exception as e:
        _log_error(f"❌ 创建预览版本失败: {e}")
        import traceback
        _log_error(traceback.format_exc())
        return None

def video_to_comfyui_video(video_path: str):
    """将视频文件转换为ComfyUI VIDEO对象 - 使用官方标准VideoFromFile，支持 MPEG-4 等特殊编码"""
    try:
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")

        _log_info(f"🎬 开始创建ComfyUI VideoFromFile对象: {video_path}")

        # 检查视频编码格式
        codec_info = None
        try:
            import subprocess
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                import json
                probe_data = json.loads(probe_result.stdout)
                if probe_data.get("streams"):
                    codec_info = probe_data["streams"][0]
                    codec_name = codec_info.get("codec_name", "unknown")
                    _log_info(f"🎬 视频编码: {codec_name}")

                    # 特别处理 MPEG-4 part 2 编码（Topaz Video AI 常用）
                    if codec_name == "mpeg4":
                        _log_info(f"🔧 检测到 MPEG-4 part 2 编码，这可能是 Topaz Video AI 处理的视频")
                        _log_info(f"💡 提示: 如果预览有问题，可以考虑转换为 H.264 格式")
        except Exception as e:
            _log_warning(f"⚠️ 无法检测视频编码: {e}")

        # 使用ComfyUI官方标准：直接从文件路径创建VideoFromFile对象
        video_obj = VideoFromFile(video_path)

        _log_info("✅ 创建ComfyUI标准VideoFromFile对象成功")

        # 测试get_dimensions方法
        try:
            dimensions = video_obj.get_dimensions()
            _log_info(f"📊 视频尺寸: {dimensions}")
        except Exception as e:
            _log_warning(f"⚠️ 无法获取视频尺寸: {e}")

        return video_obj

    except Exception as e:
        _log_error(f"创建VideoFromFile对象失败: {e}")
        return None

def extract_video_path(video):
    """从VIDEO对象提取文件路径 - 通用函数，支持多种VIDEO类型"""
    _log_info(f"🔍 尝试从VIDEO对象提取路径: {type(video)}")

    # 如果是字符串，直接返回
    if isinstance(video, str):
        _log_info(f"✅ 直接字符串路径: {video}")
        return video

    # 如果是 VHS 的 VIDEO 对象（通常是字典）
    if isinstance(video, dict):
        # VHS 的 VIDEO 对象可能包含 'filename' 和 'subfolder' 键
        if 'filename' in video:
            filename = video['filename']
            subfolder = video.get('subfolder', '')
            video_type = video.get('type', 'output')  # 'output', 'input', 'temp'

            # 构建完整路径
            if video_type == 'output':
                base_dir = folder_paths.get_output_directory()
            elif video_type == 'input':
                base_dir = folder_paths.get_input_directory()
            elif video_type == 'temp':
                base_dir = folder_paths.get_temp_directory()
            else:
                base_dir = folder_paths.get_output_directory()

            if subfolder:
                video_path = os.path.join(base_dir, subfolder, filename)
            else:
                video_path = os.path.join(base_dir, filename)

            _log_info(f"✅ 从 VHS VIDEO 字典提取路径: {video_path}")
            return video_path

    # 打印对象的所有属性（调试用）
    try:
        all_attrs = dir(video)
        _log_info(f"🔍 对象的所有属性: {[attr for attr in all_attrs if not attr.startswith('_')]}")
    except:
        pass

    # 尝试常见的文件路径属性
    path_attributes = [
        'saved_path',   # ComfyUI VideoFromFile 对象的保存路径
        'file_path',    # 我们自己的VideoFromFile对象
        'filename',     # 一些节点使用这个
        'file',         # 向后兼容
        'path',         # 通用路径属性
        'filepath',     # 文件路径
        'video_path',   # 视频路径
        'source',       # 源文件
        'url',          # URL路径
        'video_file',   # 视频文件
        'file_name',    # 文件名
    ]

    for attr in path_attributes:
        if hasattr(video, attr):
            value = getattr(video, attr)
            _log_info(f"🔍 检查属性 {attr}: {type(value)} = {value}")
            if isinstance(value, str) and value.strip() and not value.startswith('<'):
                _log_info(f"✅ 通过属性 {attr} 获取路径: {value}")
                return value.strip()

    # 如果是字典类型
    if isinstance(video, dict):
        for key in path_attributes:
            if key in video and isinstance(video[key], str) and video[key].strip():
                _log_info(f"✅ 通过字典键 {key} 获取路径: {video[key]}")
                return video[key].strip()

    # 如果是列表或元组，尝试第一个元素
    if isinstance(video, (list, tuple)) and len(video) > 0:
        first_item = video[0]
        if isinstance(first_item, str) and first_item.strip():
            _log_info(f"✅ 通过列表第一个元素获取路径: {first_item}")
            return first_item.strip()

    # 尝试 get_stream_source 方法（ComfyUI VideoFromFile 对象）
    if hasattr(video, 'get_stream_source'):
        try:
            stream_source = video.get_stream_source()
            _log_info(f"🔍 调用 get_stream_source 返回: {type(stream_source)} = {stream_source}")

            # stream_source 可能是一个对象，尝试提取路径
            if isinstance(stream_source, str):
                if stream_source.strip() and not stream_source.startswith('<'):
                    _log_info(f"✅ 通过 get_stream_source 获取路径: {stream_source}")
                    return stream_source.strip()
            elif hasattr(stream_source, 'path'):
                path = getattr(stream_source, 'path')
                if isinstance(path, str) and path.strip():
                    _log_info(f"✅ 通过 get_stream_source().path 获取路径: {path}")
                    return path.strip()
            elif hasattr(stream_source, 'file_path'):
                path = getattr(stream_source, 'file_path')
                if isinstance(path, str) and path.strip():
                    _log_info(f"✅ 通过 get_stream_source().file_path 获取路径: {path}")
                    return path.strip()
            elif hasattr(stream_source, 'filename'):
                path = getattr(stream_source, 'filename')
                if isinstance(path, str) and path.strip():
                    _log_info(f"✅ 通过 get_stream_source().filename 获取路径: {path}")
                    return path.strip()

            # 打印 stream_source 的属性
            try:
                stream_attrs = dir(stream_source)
                _log_info(f"🔍 stream_source 的属性: {[attr for attr in stream_attrs if not attr.startswith('_')]}")
            except:
                pass

        except Exception as e:
            _log_warning(f"⚠️ 调用 get_stream_source 失败: {e}")

    # 尝试调用可能的方法（但不包括 __str__，因为它可能返回对象表示）
    method_names = ['get_path', 'get_file_path', 'get_filename', 'to_string']
    for method_name in method_names:
        if hasattr(video, method_name):
            try:
                result = getattr(video, method_name)()
                _log_info(f"🔍 调用方法 {method_name} 返回: {type(result)} = {result}")
                if isinstance(result, str) and result.strip() and not result.startswith('<'):
                    _log_info(f"✅ 通过方法 {method_name} 获取路径: {result}")
                    return result.strip()
            except Exception as e:
                _log_warning(f"⚠️ 调用方法 {method_name} 失败: {e}")

    _log_error(f"❌ 无法从VIDEO对象提取路径: {type(video)}")
    _log_error(f"❌ 对象内容: {repr(video)}")
    return None

def create_video_path_wrapper(file_path):
    """创建一个视频路径包装器，用于UtilNodes兼容性"""
    # 直接返回文件路径字符串，让UtilNodes的os.path.basename()能正常工作
    return file_path

def extract_video_last_frame(video_path, output_path=None):
    """
    提取视频的最后一帧图像

    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径，如果为None则自动生成

    Returns:
        str: 输出图片的路径，失败返回None
    """
    try:
        import subprocess
        import tempfile
        from pathlib import Path

        if not os.path.exists(video_path):
            _log_error(f"视频文件不存在: {video_path}")
            return None

        # 如果没有指定输出路径，自动生成
        if output_path is None:
            video_name = Path(video_path).stem
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{video_name}_last_frame_{int(time.time())}.jpg")

        _log_info(f"🎬 正在提取视频尾帧: {video_path}")

        # 方法1：使用FFmpeg的select=eof过滤器
        cmd1 = [
            'ffmpeg',
            '-i', video_path,           # 输入视频
            '-vf', 'select=eof',        # 选择最后一帧
            '-vsync', 'vfr',            # 可变帧率
            '-frames:v', '1',           # 只输出1帧
            '-y',                       # 覆盖输出文件
            output_path
        ]

        try:
            result = subprocess.run(
                cmd1,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info(f"✅ 尾帧提取成功: {output_path}")
                return output_path
        except:
            pass

        # 方法2：如果方法1失败，使用时长计算方法
        _log_info("🔄 尝试备用方法提取尾帧...")

        # 获取视频时长
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

        if duration_result.returncode == 0:
            try:
                duration = float(duration_result.stdout.strip())
                seek_time = max(0, duration - 0.1)  # 提取最后0.1秒前的帧

                cmd2 = [
                    'ffmpeg',
                    '-ss', str(seek_time),
                    '-i', video_path,
                    '-frames:v', '1',
                    '-y',
                    output_path
                ]

                result = subprocess.run(
                    cmd2,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0 and os.path.exists(output_path):
                    _log_info(f"✅ 尾帧提取成功 (备用方法): {output_path}")
                    return output_path
            except:
                pass

        _log_error("❌ 所有尾帧提取方法都失败了")
        return None

    except Exception as e:
        _log_error(f"提取视频尾帧失败: {str(e)}")
        return None

def merge_videos_with_ffmpeg(video_paths, output_path=None):
    """使用ffmpeg合并多个视频文件"""
    try:
        import subprocess
        import tempfile

        if not video_paths or len(video_paths) < 2:
            _log_warning("⚠️ 视频数量不足，无需合并")
            return video_paths[0] if video_paths else None

        # 验证所有视频文件存在
        valid_paths = []
        for path in video_paths:
            if path and os.path.exists(path):
                valid_paths.append(path)
            else:
                _log_warning(f"⚠️ 视频文件不存在，跳过: {path}")

        if len(valid_paths) < 2:
            _log_warning("⚠️ 有效视频数量不足，无需合并")
            return valid_paths[0] if valid_paths else None

        # 生成输出文件路径 - 使用ComfyUI输出目录
        if not output_path:
            timestamp = int(time.time())
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except ImportError:
                # 推断ComfyUI输出目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                path_parts = current_dir.split(os.sep)
                comfyui_root = None

                for i in range(len(path_parts) - 1, -1, -1):
                    potential_root = os.sep.join(path_parts[:i+1])
                    if os.path.exists(os.path.join(potential_root, "main.py")):
                        comfyui_root = potential_root
                        break

                if comfyui_root:
                    output_dir = os.path.join(comfyui_root, "output")
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    import tempfile
                    output_dir = tempfile.gettempdir()
            except:
                import tempfile
                output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, f"merged_continuous_video_{timestamp}.mp4")

        _log_info(f"🎬 开始合并{len(valid_paths)}个视频文件...")
        _log_info(f"📁 输出路径: {output_path}")

        # 创建ffmpeg输入文件列表
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for path in valid_paths:
                # 使用绝对路径并转义特殊字符
                abs_path = os.path.abspath(path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
            input_list_path = f.name

        try:
            # 构建ffmpeg命令
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', input_list_path,
                '-c', 'copy',  # 直接复制流，不重新编码
                '-y',  # 覆盖输出文件
                output_path
            ]

            _log_info(f"🔧 执行ffmpeg命令: {' '.join(cmd)}")

            # 执行ffmpeg命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    _log_info(f"✅ 视频合并成功: {output_path} (大小: {file_size} bytes)")
                    return output_path
                else:
                    _log_error("❌ ffmpeg执行成功但输出文件不存在")
                    return None
            else:
                _log_error(f"❌ ffmpeg执行失败: {result.stderr}")
                return None

        finally:
            # 清理临时文件
            try:
                os.unlink(input_list_path)
            except:
                pass

    except subprocess.TimeoutExpired:
        _log_error("❌ ffmpeg执行超时")
        return None
    except FileNotFoundError:
        _log_error("❌ 未找到ffmpeg，请确保已安装ffmpeg并添加到PATH")
        return None
    except Exception as e:
        _log_error(f"❌ 视频合并失败: {str(e)}")
        return None

def get_resolution_dimensions(resolution, aspect_ratio):
    """根据分辨率和宽高比获取实际像素尺寸

    Args:
        resolution: "480p", "720p", "1080p"
        aspect_ratio: "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"

    Returns:
        tuple: (width, height) 或 None
    """
    # Seedance 1.0 pro 支持的分辨率和宽高比对应表
    resolution_map = {
        "480p": {
            "16:9": (864, 480),
            "4:3": (736, 544),
            "1:1": (640, 640),
            "3:4": (544, 736),
            "9:16": (480, 864),
            "21:9": (960, 416)
        },
        "720p": {
            "16:9": (1248, 704),
            "4:3": (1120, 832),
            "1:1": (960, 960),
            "3:4": (832, 1120),
            "9:16": (704, 1248),
            "21:9": (1504, 640)
        },
        "1080p": {
            "16:9": (1920, 1088),
            "4:3": (1664, 1248),
            "1:1": (1440, 1440),
            "3:4": (1248, 1664),
            "9:16": (1088, 1920),
            "21:9": (2176, 928)
        }
    }

    if resolution in resolution_map and aspect_ratio in resolution_map[resolution]:
        dimensions = resolution_map[resolution][aspect_ratio]
        _log_info(f"🔍 分辨率计算: {resolution} + {aspect_ratio} = {dimensions[0]}x{dimensions[1]}")
        return dimensions
    else:
        _log_warning(f"⚠️ 不支持的分辨率或宽高比组合: {resolution} + {aspect_ratio}")
        # 返回默认值
        default_dimensions = (1248, 704)  # 720p 16:9
        _log_info(f"🔧 使用默认分辨率: {default_dimensions[0]}x{default_dimensions[1]}")
        return default_dimensions

def create_blank_video_object(frames=30, height=512, width=512):
    """创建空白视频对象 - 使用临时文件创建VideoFromFile"""
    try:
        _log_info(f"🎬 创建空白视频文件: {frames}帧, {width}x{height}")

        # 创建临时视频文件
        temp_video_path = os.path.join(tempfile.gettempdir(), f"blank_video_{int(time.time())}.mp4")

        # 使用OpenCV创建空白视频文件
        if HAS_CV2:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))

            # 创建黑色帧
            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

            for _ in range(frames):
                out.write(blank_frame)

            out.release()
            _log_info(f"✅ 空白视频文件创建成功: {temp_video_path}")
        else:
            # 如果没有OpenCV，创建一个最小的MP4文件
            _log_warning("⚠️ 没有OpenCV，创建简单的空白视频对象")
            # 这里我们仍然需要一个有效的视频文件路径
            # 作为回退，我们创建一个虚拟路径
            temp_video_path = "blank_video.mp4"

        # 创建ComfyUI VideoFromFile对象
        video_obj = VideoFromFile(temp_video_path)
        return video_obj

    except Exception as e:
        _log_error(f"创建空白视频对象失败: {e}")
        # 最后的回退：创建一个简单的VideoFromFile对象
        return VideoFromFile("blank_video.mp4")


    """调用Comfly API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 调试信息
    _log_info(f"🔍 Comfly API调用:")
    _log_info(f"   - 端点: {api_url}/images/generations")
    _log_info(f"   - 模型: {payload.get('model', 'N/A')}")
    _log_info(f"   - 包含图像: {'image' in payload and bool(payload.get('image'))}")
    if 'image' in payload and payload.get('image'):
        _log_info(f"   - 图像数量: {len(payload['image'])}")
        _log_info(f"   - 第一张图像长度: {len(payload['image'][0]) if payload['image'] else 0}")

    try:
        # 使用SSL兼容的session
        session = create_ssl_compatible_session()
        response = session.post(
            f"{api_url}/images/generations",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        return response
    except Exception as e:
        _log_error(f"Comfly API调用失败: {e}")
        return None


    """调用OpenAI兼容API - 支持T8图像编辑"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        # 检查是否是T8镜像站
        if "t8star.cn" in api_url or "ai.t8star.cn" in api_url:
            # 检查是否有图像输入
            has_images = "image" in payload and payload["image"]

            # 对于T8，图生图也使用images/generations端点，而不是chat/completions
            # 只有特定的图像编辑任务才使用chat/completions
            use_chat_endpoint = False  # 暂时禁用chat端点，统一使用images/generations

            if has_images and use_chat_endpoint:
                # 图像编辑：使用chat/completions端点（暂时禁用）
                url = "https://ai.t8star.cn/v1/chat/completions"
                _log_info(f"🎨 T8图像编辑端点: {url}")

                # 构建T8图像编辑的payload格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": payload.get("prompt", "")
                            }
                        ]
                    }
                ]

                # 添加图像到消息中
                image_urls = payload.get("image", [])
                if isinstance(image_urls, str):
                    image_urls = [image_urls]

                for image_url in image_urls:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    })

                t8_payload = {
                    "model": payload.get("model", "doubao-seedream-4-0-250828"),
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7
                }

                _log_info(f"🎨 T8图像编辑请求: 模型={t8_payload.get('model')}, 消息数={len(t8_payload.get('messages', []))}")

            else:
                # 图像生成：使用images/generations端点
                url = "https://ai.t8star.cn/v1/images/generations"
                _log_info(f"🖼️ T8图像生成端点: {url}")

                # 构建T8图像生成的payload格式
                t8_payload = {
                    "prompt": payload.get("prompt", ""),
                    "model": payload.get("model", "doubao-seedream-4-0-250828"),
                    "response_format": payload.get("response_format", "url")
                }

                # 添加可选参数
                if "size" in payload:
                    t8_payload["size"] = payload["size"]
                if "n" in payload:
                    t8_payload["n"] = payload["n"]
                if "seed" in payload and payload["seed"] != -1:
                    t8_payload["seed"] = payload["seed"]
                if "watermark" in payload:
                    t8_payload["watermark"] = payload["watermark"]
                if "tail_on_partial" in payload:
                    t8_payload["tail_on_partial"] = payload["tail_on_partial"]

                # 添加图像输入支持（图生图）
                if has_images:
                    t8_payload["image"] = payload["image"]
                    _log_info(f"🖼️ T8图生图请求: 包含 {len(payload['image'])} 张输入图像")
                    _log_info(f"🔍 图像数据类型: {type(payload['image'])}")
                    if payload['image']:
                        _log_info(f"🔍 第一张图像数据长度: {len(payload['image'][0]) if payload['image'][0] else 0} 字符")

                _log_info(f"🖼️ T8图像生成请求: 模型={t8_payload.get('model')}, 提示词长度={len(t8_payload.get('prompt', ''))}")

        elif api_url.endswith('/v1/chat/completions'):
            url = api_url.replace('/v1/chat/completions', '/v1/images/generations')
            _log_info(f"🔗 转换聊天端点为图像生成端点: {url}")
            t8_payload = payload
        else:
            # 其他OpenAI兼容API
            url = f"{api_url}/v1/images/generations"
            _log_info(f"🔗 使用标准OpenAI端点: {url}")
            t8_payload = payload

        # 尝试多种连接方式解决SSL问题
        response = None
        last_error = None

        # 方法1：禁用所有SSL验证的简单方式
        try:
            import urllib3
            urllib3.disable_warnings()

            # 设置环境变量禁用SSL验证
            import os
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            os.environ['CURL_CA_BUNDLE'] = ''

            response = requests.post(
                url,
                headers=headers,
                json=t8_payload,
                timeout=timeout,
                verify=False,
                stream=False
            )
            _log_info(f"✅ 简单SSL禁用方式成功")

        except Exception as simple_error:
            last_error = simple_error
            _log_warning(f"简单SSL禁用失败: {simple_error}")

            # 方法2：使用SSL兼容的session
            try:
                session = create_ssl_compatible_session()
                response = session.post(
                    url,
                    headers=headers,
                    json=t8_payload,
                    timeout=timeout
                )
                _log_info(f"✅ SSL兼容session成功")

            except Exception as ssl_error:
                last_error = ssl_error
                _log_warning(f"SSL兼容session失败: {ssl_error}")

                # 方法3：使用curl作为备用方案
                try:
                    import subprocess
                    import tempfile

                    # 将payload写入临时文件
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(t8_payload, f)
                        temp_file = f.name

                    # 构建curl命令
                    curl_cmd = [
                        'curl', '-k', '-X', 'POST',
                        '-H', 'Content-Type: application/json',
                        '-H', f'Authorization: Bearer {headers["Authorization"].split(" ")[1]}',
                        '-d', f'@{temp_file}',
                        '--connect-timeout', '30',
                        '--max-time', str(timeout),
                        url
                    ]

                    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=timeout)

                    # 清理临时文件
                    os.unlink(temp_file)

                    if result.returncode == 0:
                        # 创建模拟response对象
                        class MockResponse:
                            def __init__(self, text, status_code=200):
                                self.text = text
                                self.status_code = status_code
                            def json(self):
                                return json.loads(self.text)

                        response = MockResponse(result.stdout)
                        _log_info(f"✅ curl备用方案成功")
                    else:
                        raise Exception(f"curl失败: {result.stderr}")

                except Exception as curl_error:
                    _log_warning(f"curl备用方案失败: {curl_error}")
                    raise last_error  # 抛出最后一个错误

        _log_info(f"🔍 T8 API响应状态: {response.status_code}")
        if response.status_code != 200:
            _log_error(f"❌ T8 API错误: {response.status_code} - {response.text}")

        return response

    except Exception as e:
        _log_error(f"OpenAI兼容API调用失败: {e}")
        return None


    """Doubao-Seedance多图参考视频生成节点"""

    @classmethod
    def INPUT_TYPES(cls):
        config = get_seedream4_config()
        mirror_sites = config.get('mirror_sites', {})
        mirror_options = list(mirror_sites.keys())

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一个美丽的场景，包含[图1]和[图2]的元素"}),
                "mirror_site": (mirror_options, {"default": mirror_options[0]}),
                "model": (["doubao-seedance-1-0-lite-i2v-250428"], {"default": "doubao-seedance-1-0-lite-i2v-250428"}),
                "duration": (["3s", "5s", "10s", "12s"], {"default": "5s"}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "16:9"}),
                "fps": ([24, 30], {"default": 30}),
                "watermark": ("BOOLEAN", {"default": False}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("video", "video_url", "response_text", "video_info", "AFVIDEO")
    FUNCTION = "generate_multi_ref_video"
    CATEGORY = "Ken-Chen/Video_Utilities"

    def __init__(self):
        self.timeout = 900  # 15分钟超时，视频生成需要更长时间
        self.max_retries = 3

    def generate_multi_ref_video(self, prompt, mirror_site, model, duration, resolution, aspect_ratio, fps, watermark=False, camera_fixed=False, api_key="", seed=-1,
                                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None):
        """生成多图参考视频"""

        # 收集参考图片
        reference_images = []
        if reference_image_1 is not None:
            reference_images.append(reference_image_1)
        if reference_image_2 is not None:
            reference_images.append(reference_image_2)
        if reference_image_3 is not None:
            reference_images.append(reference_image_3)
        if reference_image_4 is not None:
            reference_images.append(reference_image_4)

        if not reference_images:
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            return (blank_video, "", "❌ 错误：至少需要提供一张参考图片", "", blank_video_path)

        if len(reference_images) > 4:
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            return (blank_video, "", "❌ 错误：最多支持4张参考图片", "", blank_video_path)

        _log_info(f"🔍 多图参考视频生成: 参考图片数量={len(reference_images)}")

        # 获取镜像站配置
        site_config = get_mirror_site_config(mirror_site)
        api_url = site_config.get("url", "").strip()
        api_format = site_config.get("api_format", "comfly")

        # 强制修正T8镜像站的API格式（确保使用最新配置）
        if mirror_site == "t8_mirror" or "t8star.cn" in api_url:
            api_format = "volcengine"
            _log_info(f"🔧 强制修正T8镜像站API格式为: {api_format}")

        # 使用镜像站的API key（如果提供了的话）
        if site_config.get("api_key") and not api_key.strip():
            api_key = site_config["api_key"]
            _log_info(f"🔑 自动使用镜像站API Key: {api_key[:8]}...")

        if not api_key.strip():
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            return (blank_video, "", "❌ 错误：未提供API Key", "", blank_video_path)

        try:
            # 多图参考支持火山引擎格式和Comfly官方格式
            if api_format not in ["volcengine", "comfly"]:
                _log_warning(f"⚠️ 多图参考功能仅支持火山引擎和Comfly格式，当前格式: {api_format}")
                blank_video = create_blank_video_object()
                blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                return (blank_video, "", "❌ 错误：多图参考功能仅支持火山引擎官方、T8镜像站和Comfly镜像站", "", blank_video_path)

            # 构建统一的content数组格式（火山引擎和Comfly官方格式相同）
            _log_info(f"🔧 构建多图参考{api_format}格式payload")

            # 构建文本内容
            if api_format == "volcengine":
                # 火山引擎格式：使用命令行参数
                text_content = f"{prompt} --rt {aspect_ratio} --dur {duration.replace('s', '')} --fps {fps} --rs {resolution}"
                if seed != -1:
                    text_content += f" --seed {seed}"
                # 添加watermark和camera_fixed参数
                text_content += f" --wm {str(watermark).lower()} --cf {str(camera_fixed).lower()}"
                _log_info(f"🔧 火山引擎多图参考文本内容: {text_content}")
            else:  # api_format == "comfly"
                # Comfly官方格式：也使用命令行参数（与火山引擎相同）
                text_content = f"{prompt} --ratio {aspect_ratio} --dur {duration.replace('s', '')} --fps {fps} --rs {resolution}"
                if seed != -1:
                    text_content += f" --seed {seed}"
                # 添加watermark和camera_fixed参数
                text_content += f" --wm {str(watermark).lower()} --cf {str(camera_fixed).lower()}"
                _log_info(f"🔧 Comfly官方多图参考文本内容: {text_content}")

            content = [
                {
                    "type": "text",
                    "text": text_content
                }
            ]

            # 添加参考图片到content数组
            for i, ref_image in enumerate(reference_images, 1):
                _log_info(f"🔍 处理参考图片 {i}: {ref_image.shape}")

                # 统一使用完整的Data URL格式
                image_data_url = image_to_base64(ref_image, return_data_url=True)
                if image_data_url:
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }

                    # 多图参考统一使用火山引擎格式，都需要role字段
                    image_content["role"] = "reference_image"

                    content.append(image_content)
                    _log_info(f"🔧 添加参考图片{i}到content (Data URL长度: {len(image_data_url)})")
                else:
                    _log_error(f"❌ 参考图片{i} Data URL编码失败")

            # 构建payload
            payload = {
                "model": model,
                "content": content
            }

            _log_info(f"🔍 多图参考payload构建完成: 格式={api_format}, 模型={model}, content数量={len(content)}")

            # 调用多图参考视频生成API（使用火山引擎格式端点）
            response = call_multi_ref_video_api(api_url, api_key, payload, api_format, self.timeout)

            if not response or response.status_code != 200:
                blank_video = create_blank_video_object()
                blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                return (blank_video, "", "❌ 错误：视频生成任务创建失败", "", blank_video_path)

            # 从响应中提取任务ID
            try:
                result = response.json()
                task_id = None

                # 火山引擎格式的任务ID提取
                if "id" in result:
                    task_id = result["id"]
                elif "task_id" in result:
                    task_id = result["task_id"]
                elif "data" in result and isinstance(result["data"], dict):
                    if "id" in result["data"]:
                        task_id = result["data"]["id"]
                    elif "task_id" in result["data"]:
                        task_id = result["data"]["task_id"]

                if not task_id:
                    _log_error(f"❌ 无法从响应中提取任务ID: {result}")
                    blank_video = create_blank_video_object()
                    blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                    return (blank_video, "", "❌ 错误：无法获取任务ID", "", blank_video_path)

                _log_info(f"🎬 多图参考视频任务创建成功: {task_id}")

            except Exception as e:
                _log_error(f"❌ 解析任务创建响应失败: {e}")
                blank_video = create_blank_video_object()
                blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                return (blank_video, "", f"❌ 错误：解析响应失败: {str(e)}", "", blank_video_path)

            # 轮询任务状态
            max_polls = 90  # 15分钟，每10秒轮询一次
            poll_interval = 10

            for poll_count in range(1, max_polls + 1):
                _log_info(f"🔍 轮询任务状态 ({poll_count}/{max_polls})")

                status_response = call_video_task_status(api_url, api_key, task_id, api_format)
                status_result = None
                if status_response and status_response.status_code == 200:
                    status_result = status_response.json()
                    _log_info(f"🔍 任务状态响应: {str(status_result)[:200]}...")

                if status_result:
                    status = status_result.get('status', 'unknown')
                    _log_info(f"🔍 当前任务状态: '{status}' (类型: {type(status)})")

                    if status.lower() in ["completed", "success", "finished", "succeeded"] or status in ["SUCCESS", "COMPLETED", "FINISHED", "SUCCEEDED"]:
                        _log_info(f"✅ 多图参考视频生成成功")

                        # 获取视频URL - 支持多种响应格式
                        video_url = None

                        # 格式1: data.content.video_url (Comfly多图参考格式)
                        if 'data' in status_result and isinstance(status_result['data'], dict):
                            if 'content' in status_result['data'] and isinstance(status_result['data']['content'], dict):
                                if 'video_url' in status_result['data']['content']:
                                    video_url = status_result['data']['content']['video_url']

                        # 格式2: content.video_url (火山引擎多图参考格式)
                        if not video_url and 'content' in status_result and isinstance(status_result['content'], dict):
                            if 'video_url' in status_result['content']:
                                video_url = status_result['content']['video_url']

                        # 格式3: video_result数组格式
                        if not video_url and 'video_result' in status_result and status_result['video_result']:
                            video_result = status_result['video_result'][0] if isinstance(status_result['video_result'], list) else status_result['video_result']
                            video_url = video_result.get('url')

                        # 格式4: 直接video_url字段
                        if not video_url and 'video_url' in status_result:
                            video_url = status_result['video_url']

                        # 格式4: result.video_url格式
                        if not video_url and 'result' in status_result and status_result['result']:
                            result = status_result['result']
                            if isinstance(result, dict) and 'video_url' in result:
                                video_url = result['video_url']
                                _log_info(f"🔍 从result.video_url提取视频URL")

                        if video_url:
                            _log_info(f"🎬 获取到视频URL: {video_url}")

                            # 下载并转换视频
                            video_path = download_video_from_url(video_url)
                            if video_path:
                                _log_info(f"🎬 开始转换视频为ComfyUI对象...")
                                video_obj = video_to_comfyui_video(video_path)
                                if video_obj is not None:
                                    video_info = f"模型: {model}, 参考图片: {len(reference_images)}张, 时长: {duration}, 分辨率: {resolution}, 宽高比: {aspect_ratio}, 帧率: {fps}fps, 任务ID: {task_id}"
                                    return (video_obj, video_url, "✅ 多图参考视频生成成功", video_info, video_path)
                                else:
                                    _log_error("❌ 视频转换失败")
                                    blank_video = create_blank_video_object()
                                    blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                                    return (blank_video, video_url, "❌ 视频转换失败", "", blank_video_path)
                            else:
                                _log_error("❌ 视频下载失败")
                                blank_video = create_blank_video_object()
                                blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                                return (blank_video, video_url, "❌ 视频下载失败", "", blank_video_path)
                        else:
                            _log_error("❌ 未获取到视频URL")
                            blank_video = create_blank_video_object()
                            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                            return (blank_video, "", "❌ 未获取到视频URL", "", blank_video_path)

                    elif status.lower() in ["failed", "error"] or status in ["FAILED", "ERROR"]:
                        fail_reason = status_result.get('fail_reason', '未知错误')
                        _log_error(f"❌ 多图参考视频生成失败: {fail_reason}")
                        blank_video = create_blank_video_object()
                        blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
                        return (blank_video, "", f"❌ 视频生成失败: {fail_reason}", "", blank_video_path)

                    elif status.lower() in ["running", "processing", "in_progress", "not_start", "queued"] or status in ["RUNNING", "PROCESSING", "IN_PROGRESS", "NOT_START", "QUEUED"]:
                        _log_info(f"⏳ 任务进行中，状态: {status}")
                        if poll_count < max_polls:
                            time.sleep(poll_interval)
                        continue
                    else:
                        _log_warning(f"⚠️ 未知任务状态: {status}")
                        if poll_count < max_polls:
                            time.sleep(poll_interval)
                        continue
                else:
                    _log_warning(f"⚠️ 无法获取任务状态，响应: {status_response}")
                    if poll_count < max_polls:
                        time.sleep(poll_interval)
                    continue

            # 超时处理
            _log_error(f"❌ 多图参考视频生成超时")
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            return (blank_video, "", "❌ 视频生成超时", "", blank_video_path)

        except Exception as e:
            _log_error(f"❌ 多图参考视频生成异常: {e}")
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            return (blank_video, "", f"❌ 错误：{str(e)}", "", blank_video_path)

class VideoStitchingNode:
    """视频拼接节点 - 最多可以将8个视频拼接在一起"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("VIDEO",),
            },
            "optional": {
                "video2": ("VIDEO",),
                "video3": ("VIDEO",),
                "video4": ("VIDEO",),
                "video5": ("VIDEO",),
                "video6": ("VIDEO",),
                "video7": ("VIDEO",),
                "video8": ("VIDEO",),
                "output_filename": ("STRING", {"default": ""}),
                "stitch_method": (["concat", "concat_crossfade", "concat_advanced", "concat_morph", "concat_optical_flow", "hstack", "vstack", "grid2x2", "grid2x3", "grid2x4"], {"default": "concat"}),
                "output_quality": (["high", "medium", "low"], {"default": "high"}),
                "scale_videos": ("BOOLEAN", {"default": True}),
                "smooth_transitions": ("BOOLEAN", {"default": True}),
                "transition_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "force_keyframes": ("BOOLEAN", {"default": True}),
                "transition_type": (["fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown", "smoothleft", "smoothright", "smoothup", "smoothdown", "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose", "dissolve", "pixelize", "radial", "smoothradial"], {"default": "fade"}),
                "motion_compensation": ("BOOLEAN", {"default": False}),
                "edge_enhancement": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "VIDEO")
    RETURN_NAMES = ("stitched_video", "video_path", "AFVIDEO")
    FUNCTION = "stitch_videos"
    CATEGORY = "Ken-Chen/Video_Utilities"

    def __init__(self):
        self.timeout = 300  # 5分钟超时，视频处理需要更长时间

    def stitch_videos(self, video1, video2=None, video3=None, video4=None, video5=None, video6=None, video7=None, video8=None,
                     output_filename="", stitch_method="concat", output_quality="high", scale_videos=True,
                     smooth_transitions=True, transition_duration=0.5, force_keyframes=True, transition_type="fade",
                     motion_compensation=False, edge_enhancement=False):
        """
        拼接多个视频

        Args:
            video1-video8: ComfyUI VIDEO对象
            output_filename: 输出文件名（可选）
            stitch_method: 拼接方法
            output_quality: 输出质量
            scale_videos: 是否缩放视频到统一尺寸

        Returns:
            tuple: (拼接后的VIDEO对象, 视频文件路径)
        """
        try:
            _log_info("🎬 开始视频拼接...")

            # 收集所有有效的视频
            videos = [video1]
            for video in [video2, video3, video4, video5, video6, video7, video8]:
                if video is not None:
                    videos.append(video)

            if len(videos) < 2:
                error_msg = "至少需要2个视频才能进行拼接"
                _log_error(error_msg)
                return self._create_error_result(error_msg)

            _log_info(f"📊 将拼接{len(videos)}个视频，使用{stitch_method}方法")

            # 获取视频文件路径
            video_paths = []
            for i, video in enumerate(videos):
                _log_info(f"🔍 处理第{i+1}个视频...")
                video_path = self._extract_video_path(video)
                if not video_path:
                    error_msg = f"无法获取第{i+1}个视频的有效路径: {video_path}"
                    _log_error(error_msg)
                    _log_error(f"视频对象详情: type={type(video)}, repr={repr(video)}")
                    return self._create_error_result(error_msg)

                if not os.path.exists(video_path):
                    error_msg = f"第{i+1}个视频文件不存在: {video_path}"
                    _log_error(error_msg)
                    return self._create_error_result(error_msg)

                video_paths.append(video_path)
                _log_info(f"✅ 第{i+1}个视频: {video_path}")

            # 生成输出文件路径
            if not output_filename:
                output_filename = f"stitched_video_{stitch_method}_{int(time.time())}.mp4"

            if not output_filename.lower().endswith('.mp4'):
                output_filename += '.mp4'

            # 使用ComfyUI的输出目录而不是系统临时目录
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                _log_info(f"📁 使用ComfyUI输出目录: {output_dir}")
            except ImportError:
                # 如果在ComfyUI环境外，尝试推断ComfyUI路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                comfyui_root = None

                # 向上查找ComfyUI根目录
                path_parts = current_dir.split(os.sep)
                for i in range(len(path_parts) - 1, -1, -1):
                    potential_root = os.sep.join(path_parts[:i+1])
                    if os.path.exists(os.path.join(potential_root, "main.py")) and \
                       os.path.exists(os.path.join(potential_root, "nodes.py")):
                        comfyui_root = potential_root
                        break

                if comfyui_root:
                    output_dir = os.path.join(comfyui_root, "output")
                    os.makedirs(output_dir, exist_ok=True)
                    _log_info(f"📁 推断ComfyUI输出目录: {output_dir}")
                else:
                    import tempfile
                    output_dir = tempfile.gettempdir()
                    _log_info(f"📁 回退到系统临时目录: {output_dir}")
            except Exception as e:
                import tempfile
                output_dir = tempfile.gettempdir()
                _log_info(f"📁 异常，使用系统临时目录: {output_dir} (错误: {e})")

            output_path = os.path.join(output_dir, output_filename)

            # 根据拼接方法执行不同的处理
            success = False
            if stitch_method == "concat":
                success = self._concat_videos(video_paths, output_path, output_quality, smooth_transitions, transition_duration, force_keyframes)
            elif stitch_method == "concat_crossfade":
                if len(video_paths) <= 2:
                    success = self._concat_with_crossfade_transitions(video_paths, output_path, output_quality, transition_duration)
                else:
                    success = self._concat_with_xfade_multiple(video_paths, output_path, output_quality, transition_duration)
            elif stitch_method == "concat_advanced":
                success = self._concat_with_advanced_transitions(video_paths, output_path, output_quality, transition_duration, transition_type, motion_compensation, edge_enhancement)
            elif stitch_method == "concat_morph":
                success = self._concat_with_morphing_transitions(video_paths, output_path, output_quality, transition_duration, motion_compensation)
            elif stitch_method == "concat_optical_flow":
                success = self._concat_with_optical_flow_transitions(video_paths, output_path, output_quality, transition_duration)
            elif stitch_method == "hstack":
                success = self._hstack_videos(video_paths, output_path, output_quality, scale_videos)
            elif stitch_method == "vstack":
                success = self._vstack_videos(video_paths, output_path, output_quality, scale_videos)
            elif stitch_method == "grid2x2":
                success = self._grid_videos(video_paths, output_path, output_quality, "2x2", scale_videos)
            elif stitch_method == "grid2x3":
                success = self._grid_videos(video_paths, output_path, output_quality, "2x3", scale_videos)
            elif stitch_method == "grid2x4":
                success = self._grid_videos(video_paths, output_path, output_quality, "2x4", scale_videos)

            if not success:
                error_msg = f"视频拼接失败，方法: {stitch_method}"
                _log_error(error_msg)
                return self._create_error_result(error_msg)

            # 转换为ComfyUI VIDEO对象
            stitched_video = video_to_comfyui_video(output_path)
            if stitched_video:
                stitched_video.file_path = output_path
                _log_info(f"✅ 视频拼接成功: {output_path}")

                # AFVIDEO使用路径包装器，与标准视频节点保持一致
                afvideo = create_video_path_wrapper(output_path) if output_path else create_blank_video_object()

                return (stitched_video, output_path, afvideo)
            else:
                error_msg = "拼接视频转换为ComfyUI对象失败"
                _log_error(error_msg)
                return self._create_error_result(error_msg)

        except Exception as e:
            error_msg = f"视频拼接失败: {str(e)}"
            _log_error(error_msg)
            return self._create_error_result(error_msg)

    def _extract_video_path(self, video):
        """从VIDEO对象提取文件路径"""
        _log_info(f"🔍 尝试从VIDEO对象提取路径: {type(video)}")

        # 如果是字符串，直接返回
        if isinstance(video, str):
            _log_info(f"✅ 直接字符串路径: {video}")
            return video

        # 尝试常见的文件路径属性
        path_attributes = [
            'file_path',    # 我们自己的VideoFromFile对象
            'filename',     # 一些节点使用这个
            'file',         # 向后兼容
            'path',         # 通用路径属性
            'filepath',     # 文件路径
            'video_path',   # 视频路径
            'source',       # 源文件
            'url',          # URL路径
            'video_file',   # 视频文件
            'file_name',    # 文件名
        ]

        for attr in path_attributes:
            if hasattr(video, attr):
                value = getattr(video, attr)
                if value and isinstance(value, str):
                    _log_info(f"✅ 从属性 {attr} 获取路径: {value}")
                    return value
                elif value:
                    _log_info(f"⚠️ 属性 {attr} 存在但不是字符串: {type(value)} = {value}")

        # 如果是字典类型，尝试从字典中获取路径
        if isinstance(video, dict):
            for key in ['file_path', 'filename', 'path', 'url', 'source']:
                if key in video and isinstance(video[key], str):
                    _log_info(f"✅ 从字典键 {key} 获取路径: {video[key]}")
                    return video[key]

        # 如果有__dict__属性，打印所有属性用于调试
        if hasattr(video, '__dict__'):
            _log_info(f"🔍 VIDEO对象属性: {list(video.__dict__.keys())}")
            for key, value in video.__dict__.items():
                if isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower() or 'url' in key.lower()):
                    _log_info(f"✅ 从__dict__属性 {key} 获取路径: {value}")
                    return value

        # 最后尝试：如果对象可以转换为字符串且看起来像路径
        try:
            str_repr = str(video)
            if str_repr and ('/' in str_repr or '\\' in str_repr or str_repr.endswith('.mp4')):
                _log_info(f"✅ 从字符串表示获取路径: {str_repr}")
                return str_repr
        except:
            pass

        _log_error(f"❌ 无法从VIDEO对象提取路径，对象类型: {type(video)}")
        return None

    def _get_quality_params(self, quality):
        """获取质量参数"""
        quality_settings = {
            "high": ["-crf", "18", "-preset", "medium"],
            "medium": ["-crf", "23", "-preset", "fast"],
            "low": ["-crf", "28", "-preset", "faster"]
        }
        return quality_settings.get(quality, quality_settings["high"])

    def _concat_videos(self, video_paths, output_path, quality, smooth_transitions=True, transition_duration=0.5, force_keyframes=True):
        """连续拼接视频（时间轴上连接）- 改进版本减少闪烁"""
        try:
            import subprocess
            import tempfile

            _log_info("🔗 使用改进的concat方法拼接视频...")

            # 首先检查视频属性一致性
            video_info = self._analyze_video_properties(video_paths)
            if not video_info:
                _log_error("无法分析视频属性")
                return False

            # 创建concat文件列表 - 使用合适的临时目录
            try:
                import folder_paths
                temp_dir = folder_paths.get_temp_directory()
            except:
                import tempfile
                temp_dir = tempfile.gettempdir()
            concat_file = os.path.join(temp_dir, f"concat_list_{int(time.time())}.txt")

            with open(concat_file, 'w', encoding='utf-8') as f:
                for video_path in video_paths:
                    # 使用绝对路径并转义特殊字符
                    abs_path = os.path.abspath(video_path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")

            # 根据视频属性一致性选择处理方式
            if video_info['consistent']:
                # 属性一致，尝试直接复制流（最快，无质量损失）
                success = self._concat_with_copy(concat_file, output_path)
                if success:
                    self._cleanup_temp_file(concat_file)
                    return True
                _log_info("🔄 直接复制失败，尝试重新编码...")

            # 属性不一致或直接复制失败，使用改进的重新编码方法
            success = self._concat_with_smooth_transitions(concat_file, output_path, quality, video_info, smooth_transitions, transition_duration, force_keyframes)

            self._cleanup_temp_file(concat_file)
            return success

        except Exception as e:
            _log_error(f"concat拼接失败: {str(e)}")
            return False

    def _analyze_video_properties(self, video_paths):
        """分析视频属性，检查一致性"""
        try:
            import subprocess
            import json

            _log_info("🔍 分析视频属性...")

            video_props = []
            for video_path in video_paths:
                # 使用ffprobe获取视频信息
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_streams',
                    '-select_streams', 'v:0',
                    video_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    _log_error(f"无法获取视频信息: {video_path}")
                    return None

                try:
                    info = json.loads(result.stdout)
                    if 'streams' in info and len(info['streams']) > 0:
                        stream = info['streams'][0]
                        props = {
                            'width': stream.get('width', 0),
                            'height': stream.get('height', 0),
                            'fps': eval(stream.get('r_frame_rate', '0/1')),
                            'codec': stream.get('codec_name', ''),
                            'pix_fmt': stream.get('pix_fmt', '')
                        }
                        video_props.append(props)
                        _log_info(f"📊 {os.path.basename(video_path)}: {props['width']}x{props['height']} @{props['fps']:.2f}fps {props['codec']}")
                    else:
                        _log_error(f"无法解析视频流信息: {video_path}")
                        return None
                except json.JSONDecodeError:
                    _log_error(f"无法解析ffprobe输出: {video_path}")
                    return None

            # 检查属性一致性
            if not video_props:
                return None

            first_props = video_props[0]
            consistent = all(
                props['width'] == first_props['width'] and
                props['height'] == first_props['height'] and
                abs(props['fps'] - first_props['fps']) < 0.1 and
                props['codec'] == first_props['codec'] and
                props['pix_fmt'] == first_props['pix_fmt']
                for props in video_props
            )

            _log_info(f"✅ 视频属性一致性: {'是' if consistent else '否'}")

            return {
                'consistent': consistent,
                'properties': video_props,
                'target_width': first_props['width'],
                'target_height': first_props['height'],
                'target_fps': first_props['fps'],
                'target_codec': first_props['codec'],
                'target_pix_fmt': first_props['pix_fmt']
            }

        except Exception as e:
            _log_error(f"分析视频属性失败: {str(e)}")
            return None

    def _concat_with_copy(self, concat_file, output_path):
        """使用流复制方式拼接（最快，适用于属性一致的视频）"""
        try:
            import subprocess

            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # 直接复制流
                '-avoid_negative_ts', 'make_zero',  # 避免负时间戳
                '-y',
                output_path
            ]

            _log_info(f"🔧 执行流复制命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 流复制拼接成功")
                return True
            else:
                _log_info(f"⚠️ 流复制失败: {result.stderr}")
                return False

        except Exception as e:
            _log_error(f"流复制拼接失败: {str(e)}")
            return False

    def _concat_with_smooth_transitions(self, concat_file, output_path, quality, video_info, smooth_transitions=True, transition_duration=0.5, force_keyframes=True):
        """使用平滑过渡的重新编码方式拼接"""
        try:
            import subprocess

            quality_params = self._get_quality_params(quality)

            # 构建改进的FFmpeg命令，减少闪烁
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',  # 强制使用H.264编码器
                '-pix_fmt', 'yuv420p',  # 统一像素格式
                '-r', str(int(video_info['target_fps'])),  # 统一帧率
                '-s', f"{video_info['target_width']}x{video_info['target_height']}",  # 统一分辨率
                '-vsync', 'cfr',  # 恒定帧率
                '-bf', '2',  # B帧数量
                '-sc_threshold', '0',  # 禁用场景切换检测
                '-avoid_negative_ts', 'make_zero',  # 避免负时间戳
                '-fflags', '+genpts',  # 生成PTS
            ]

            # 根据参数添加关键帧控制
            if force_keyframes:
                keyframe_interval = max(1, int(video_info['target_fps'] * 2))  # 每2秒一个关键帧
                cmd.extend([
                    '-force_key_frames', f'expr:gte(t,n_forced*2)',
                    '-g', str(keyframe_interval),
                ])

            # 添加平滑过渡滤镜（如果启用）
            if smooth_transitions and transition_duration > 0:
                # 使用minterpolate滤镜进行帧插值，减少跳跃感
                cmd.extend([
                    '-vf', f'minterpolate=fps={video_info["target_fps"]}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'
                ])

            cmd.extend(quality_params + ['-y', output_path])

            _log_info(f"🔧 执行平滑过渡编码: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 平滑过渡拼接成功")
                return True
            else:
                _log_error(f"❌ 平滑过渡拼接失败: {result.stderr}")
                # 如果高级滤镜失败，尝试基础方法
                if smooth_transitions:
                    _log_info("🔄 尝试基础平滑方法...")
                    return self._concat_with_basic_smooth(concat_file, output_path, quality, video_info)
                return False

        except Exception as e:
            _log_error(f"平滑过渡拼接失败: {str(e)}")
            return False

    def _concat_with_basic_smooth(self, concat_file, output_path, quality, video_info):
        """基础平滑拼接方法（备用）"""
        try:
            import subprocess

            quality_params = self._get_quality_params(quality)

            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(int(video_info['target_fps'])),
                '-s', f"{video_info['target_width']}x{video_info['target_height']}",
                '-vsync', 'cfr',
                '-bf', '2',
                '-g', str(int(video_info['target_fps'] * 2)),
                '-sc_threshold', '0',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
            ] + quality_params + [
                '-y',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            _log_error(f"基础平滑拼接失败: {str(e)}")
            return False

    def _cleanup_temp_file(self, temp_file):
        """清理临时文件"""
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            _log_error(f"清理临时文件失败: {str(e)}")

    def _concat_with_crossfade_transitions(self, video_paths, output_path, quality, transition_duration=0.5):
        """使用交叉淡化过渡效果拼接视频"""
        try:
            import subprocess

            if len(video_paths) < 2:
                return self._concat_videos(video_paths, output_path, quality, False, 0, True)

            _log_info(f"🎬 使用交叉淡化过渡拼接 {len(video_paths)} 个视频...")

            # 对于交叉淡化，我们使用更简单但有效的方法：
            # 1. 先用concat正常拼接
            # 2. 然后在拼接点添加淡化效果

            # 首先获取视频信息以计算总时长
            video_info = self._analyze_video_properties(video_paths)
            if not video_info:
                _log_error("无法分析视频属性，回退到普通拼接")
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            # 计算每个视频的时长和累积时长
            video_durations = []
            cumulative_time = 0

            for video_path in video_paths:
                try:
                    cmd_duration = [
                        'ffprobe',
                        '-v', 'quiet',
                        '-show_entries', 'format=duration',
                        '-of', 'csv=p=0',
                        video_path
                    ]
                    result = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        duration = float(result.stdout.strip())
                        video_durations.append(duration)
                        cumulative_time += duration
                    else:
                        video_durations.append(2.0)  # 默认2秒
                        cumulative_time += 2.0
                except:
                    video_durations.append(2.0)
                    cumulative_time += 2.0

            _log_info(f"📊 视频时长: {[f'{d:.1f}s' for d in video_durations]}, 总时长: {cumulative_time:.1f}s")

            # 构建输入
            inputs = []
            for video_path in video_paths:
                inputs.extend(['-i', video_path])

            # 构建简化的交叉淡化滤镜
            if len(video_paths) == 2:
                # 两个视频的简单交叉淡化
                filter_complex = self._build_simple_crossfade_filter(video_durations, transition_duration)
            else:
                # 多个视频使用改进的concat方法
                _log_info("🔄 多视频交叉淡化，使用改进的concat方法...")
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            quality_params = self._get_quality_params(quality)

            cmd = [
                'ffmpeg'
            ] + inputs + [
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

            _log_info(f"🔧 执行交叉淡化命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 交叉淡化拼接成功")
                return True
            else:
                _log_error(f"❌ 交叉淡化拼接失败: {result.stderr}")
                # 回退到普通拼接
                _log_info("🔄 回退到普通拼接方法...")
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

        except Exception as e:
            _log_error(f"交叉淡化拼接失败: {str(e)}")
            # 回退到普通拼接
            return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

    def _build_simple_crossfade_filter(self, video_durations, transition_duration):
        """构建简单的两视频交叉淡化滤镜"""
        if len(video_durations) != 2:
            return "[0:v][1:v]concat=n=2:v=1[output]"

        duration1, duration2 = video_durations

        # 确保过渡时间不超过较短视频的一半
        max_transition = min(duration1, duration2) / 2
        actual_transition = min(transition_duration, max_transition)

        if actual_transition <= 0:
            return "[0:v][1:v]concat=n=2:v=1[output]"

        # 使用xfade滤镜进行交叉淡化（更专业的方法）
        # xfade滤镜会自动处理时间对齐
        offset_time = duration1 - actual_transition

        filter_complex = f"[0:v][1:v]xfade=transition=fade:duration={actual_transition}:offset={offset_time}[output]"

        return filter_complex

    def _concat_with_xfade_multiple(self, video_paths, output_path, quality, transition_duration=0.5):
        """使用xfade滤镜拼接多个视频（改进版本）"""
        try:
            import subprocess
            import tempfile

            _log_info(f"🎬 使用xfade滤镜拼接 {len(video_paths)} 个视频...")

            if len(video_paths) == 2:
                # 两个视频直接使用xfade
                return self._concat_with_crossfade_transitions(video_paths, output_path, quality, transition_duration)

            # 多个视频需要递归处理
            temp_dir = tempfile.mkdtemp()
            intermediate_files = []

            try:
                current_video = video_paths[0]

                for i in range(1, len(video_paths)):
                    next_video = video_paths[i]
                    temp_output = os.path.join(temp_dir, f"intermediate_{i}.mp4")

                    # 使用两视频交叉淡化
                    success = self._concat_with_crossfade_transitions(
                        [current_video, next_video],
                        temp_output,
                        quality,
                        transition_duration
                    )

                    if not success:
                        _log_error(f"中间步骤 {i} 失败")
                        return False

                    intermediate_files.append(temp_output)
                    current_video = temp_output

                # 复制最终结果
                if intermediate_files:
                    final_temp = intermediate_files[-1]
                    if os.path.exists(final_temp):
                        import shutil
                        shutil.copy2(final_temp, output_path)
                        return os.path.exists(output_path)

                return False

            finally:
                # 清理临时文件
                for temp_file in intermediate_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass

        except Exception as e:
            _log_error(f"多视频xfade拼接失败: {str(e)}")
            return False

    def _concat_with_advanced_transitions(self, video_paths, output_path, quality, transition_duration=0.5, transition_type="fade", motion_compensation=False, edge_enhancement=False):
        """使用高级过渡效果拼接视频"""
        try:
            import subprocess

            if len(video_paths) < 2:
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            _log_info(f"🎨 使用高级过渡效果拼接 {len(video_paths)} 个视频，过渡类型: {transition_type}")

            # 获取视频信息
            video_info = self._analyze_video_properties(video_paths)
            if not video_info:
                _log_error("无法分析视频属性，回退到普通拼接")
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            # 构建输入
            inputs = []
            for video_path in video_paths:
                inputs.extend(['-i', video_path])

            # 构建高级过渡滤镜
            if len(video_paths) == 2:
                filter_complex = self._build_advanced_transition_filter(video_paths, transition_duration, transition_type, motion_compensation, edge_enhancement)
            else:
                # 多视频使用递归处理
                return self._concat_advanced_multiple(video_paths, output_path, quality, transition_duration, transition_type, motion_compensation, edge_enhancement)

            quality_params = self._get_quality_params(quality)

            cmd = [
                'ffmpeg'
            ] + inputs + [
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

            _log_info(f"🔧 执行高级过渡命令...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 2  # 高级处理需要更多时间
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 高级过渡拼接成功")
                return True
            else:
                _log_error(f"❌ 高级过渡拼接失败")
                # 回退到交叉淡化
                _log_info("🔄 回退到交叉淡化方法...")
                return self._concat_with_crossfade_transitions(video_paths, output_path, quality, transition_duration)

        except Exception as e:
            _log_error(f"高级过渡拼接失败: {str(e)}")
            # 回退到交叉淡化
            return self._concat_with_crossfade_transitions(video_paths, output_path, quality, transition_duration)

    def _build_advanced_transition_filter(self, video_paths, transition_duration, transition_type, motion_compensation, edge_enhancement):
        """构建高级过渡滤镜"""
        try:
            # 获取视频时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        durations.append(float(result.stdout.strip()))
                    else:
                        durations.append(4.0)  # 默认4秒
                except:
                    durations.append(4.0)

            if len(durations) < 2:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            duration1, duration2 = durations[0], durations[1]
            max_transition = min(duration1, duration2) / 2
            actual_transition = min(transition_duration, max_transition)

            if actual_transition <= 0:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            offset_time = duration1 - actual_transition

            # 预处理滤镜
            preprocess_filters = []

            # 边缘增强
            if edge_enhancement:
                preprocess_filters.extend([
                    "[0:v]unsharp=5:5:1.0:5:5:0.0[v0enhanced]",
                    "[1:v]unsharp=5:5:1.0:5:5:0.0[v1enhanced]"
                ])
                input_labels = ["[v0enhanced]", "[v1enhanced]"]
            else:
                input_labels = ["[0:v]", "[1:v]"]

            # 运动补偿（使用minterpolate进行帧插值）
            if motion_compensation:
                preprocess_filters.extend([
                    f"{input_labels[0]}minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v0smooth]",
                    f"{input_labels[1]}minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v1smooth]"
                ])
                input_labels = ["[v0smooth]", "[v1smooth]"]

            # 构建xfade过渡
            xfade_filter = f"{input_labels[0]}{input_labels[1]}xfade=transition={transition_type}:duration={actual_transition}:offset={offset_time}[output]"

            # 组合所有滤镜
            if preprocess_filters:
                filter_complex = ";".join(preprocess_filters) + ";" + xfade_filter
            else:
                filter_complex = xfade_filter

            return filter_complex

        except Exception as e:
            _log_error(f"构建高级过渡滤镜失败: {str(e)}")
            return f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={max(0, durations[0] - transition_duration)}[output]"

    def _concat_advanced_multiple(self, video_paths, output_path, quality, transition_duration, transition_type, motion_compensation, edge_enhancement):
        """多视频高级过渡拼接 - 修复时长计算问题"""
        try:
            # 对于多视频，使用一次性滤镜链而不是递归拼接
            # 这样可以避免重复减去过渡时间的问题

            if len(video_paths) == 2:
                # 两个视频直接使用原方法
                return self._concat_with_advanced_transitions(
                    video_paths, output_path, quality, transition_duration,
                    transition_type, motion_compensation, edge_enhancement
                )

            # 多于2个视频时，构建一次性滤镜链
            return self._concat_advanced_multiple_chain(
                video_paths, output_path, quality, transition_duration,
                transition_type, motion_compensation, edge_enhancement
            )

        except Exception as e:
            _log_error(f"多视频高级过渡拼接失败: {str(e)}")
            return False

    def _concat_advanced_multiple_chain(self, video_paths, output_path, quality, transition_duration, transition_type, motion_compensation, edge_enhancement):
        """使用一次性滤镜链拼接多个视频 - 正确的时长计算"""
        try:
            import subprocess

            # 获取所有视频的时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        durations.append(float(result.stdout.strip()))
                    else:
                        durations.append(4.0)  # 默认4秒
                except:
                    durations.append(4.0)

            if len(durations) < 2:
                return False

            # 计算过渡参数
            min_duration = min(durations)
            max_transition = min_duration / 2
            actual_transition = min(transition_duration, max_transition)

            if actual_transition <= 0:
                # 无过渡，使用简单concat
                return self._simple_concat_multiple(video_paths, output_path)

            # 构建多视频xfade滤镜链
            filter_complex = self._build_multiple_xfade_chain(video_paths, durations, actual_transition, transition_type)

            if not filter_complex:
                _log_error("构建多视频滤镜链失败")
                return False

            # 执行FFmpeg命令
            cmd = ['ffmpeg']

            # 添加输入文件
            for video_path in video_paths:
                cmd.extend(['-i', video_path])

            # 添加滤镜和输出参数
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[output]',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',
                '-crf', '23',
                '-y',
                output_path
            ])

            _log_info(f"🔧 执行多视频高级过渡命令...")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 多视频高级过渡拼接成功")
                return True
            else:
                _log_error(f"多视频高级过渡拼接失败: {result.stderr}")
                return False

        except Exception as e:
            _log_error(f"多视频高级过渡拼接异常: {str(e)}")
            return False

    def _build_multiple_xfade_chain(self, video_paths, durations, transition_duration, transition_type):
        """构建多视频xfade滤镜链"""
        try:
            if len(video_paths) < 2:
                return None

            if len(video_paths) == 2:
                # 两个视频的简单情况
                offset_time = durations[0] - transition_duration
                return f"[0:v][1:v]xfade=transition={transition_type}:duration={transition_duration}:offset={offset_time}[output]"

            # 多个视频的复杂情况
            filter_parts = []
            current_offset = 0

            # 第一个过渡
            offset_time = durations[0] - transition_duration
            filter_parts.append(f"[0:v][1:v]xfade=transition={transition_type}:duration={transition_duration}:offset={offset_time}[v01]")
            current_offset = durations[0] + durations[1] - transition_duration

            # 后续过渡
            for i in range(2, len(video_paths)):
                input_label = f"v0{i-1}" if i == 2 else f"v0{i-1}"
                output_label = f"v0{i}" if i < len(video_paths) - 1 else "output"

                # 计算这个过渡的偏移时间
                offset_time = current_offset - transition_duration
                filter_parts.append(f"[{input_label}][{i}:v]xfade=transition={transition_type}:duration={transition_duration}:offset={offset_time}[{output_label}]")

                current_offset += durations[i] - transition_duration

            return ";".join(filter_parts)

        except Exception as e:
            _log_error(f"构建多视频滤镜链失败: {str(e)}")
            return None

    def _simple_concat_multiple(self, video_paths, output_path):
        """简单的多视频拼接（无过渡）"""
        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for video_path in video_paths:
                    f.write(f"file '{video_path}'\n")
                concat_file = f.name

            try:
                cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',
                    '-y',
                    output_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0 and os.path.exists(output_path):
                    return True
                else:
                    return False

            finally:
                try:
                    os.unlink(concat_file)
                except:
                    pass

        except Exception as e:
            _log_error(f"简单多视频拼接失败: {str(e)}")
            return False

    def _concat_with_morphing_transitions(self, video_paths, output_path, quality, transition_duration=0.5, motion_compensation=False):
        """使用形态学过渡拼接视频（实验性）"""
        try:
            import subprocess

            if len(video_paths) < 2:
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            _log_info(f"🧬 使用形态学过渡拼接 {len(video_paths)} 个视频...")

            # 获取视频信息
            video_info = self._analyze_video_properties(video_paths)
            if not video_info:
                _log_error("无法分析视频属性，回退到高级过渡")
                return self._concat_with_advanced_transitions(video_paths, output_path, quality, transition_duration, "fade", motion_compensation, False)

            # 对于形态学过渡，我们使用blend滤镜和morphological操作
            if len(video_paths) == 2:
                filter_complex = self._build_morphing_filter(video_paths, transition_duration, motion_compensation)
            else:
                # 多视频使用一次性滤镜链，避免时长计算错误
                return self._concat_morphing_multiple_chain(video_paths, output_path, quality, transition_duration, motion_compensation)

            inputs = []
            for video_path in video_paths:
                inputs.extend(['-i', video_path])

            quality_params = self._get_quality_params(quality)

            cmd = [
                'ffmpeg'
            ] + inputs + [
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

            _log_info(f"🔧 执行形态学过渡命令...")

            # 大幅缩短超时时间 - 形态学过渡也应该快速处理
            video_info = self._analyze_video_properties(video_paths)
            base_timeout = 20  # 基础20秒超时

            if video_info and 'target_width' in video_info and 'target_height' in video_info:
                pixels = video_info['target_width'] * video_info['target_height']
                if pixels > 1000000:  # 大于1MP (如1248x704)
                    timeout_seconds = 45  # 最多45秒
                elif pixels > 500000:  # 大于0.5MP
                    timeout_seconds = 30  # 30秒
                else:
                    timeout_seconds = 20  # 20秒
            else:
                timeout_seconds = 20

            _log_info(f"⏱️ 形态学过渡超时设置: {timeout_seconds}秒 (快速处理策略)")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 形态学过渡拼接成功")
                return True
            else:
                _log_error(f"❌ 形态学过渡拼接失败")
                if result.stderr:
                    _log_error(f"FFmpeg错误: {result.stderr[:300]}...")
                # 回退到高级过渡
                _log_info("🔄 回退到高级过渡方法...")
                return self._concat_with_advanced_transitions(video_paths, output_path, quality, transition_duration, "dissolve", motion_compensation, True)

        except subprocess.TimeoutExpired:
            _log_error(f"⏰ 形态学过渡超时 ({timeout_seconds}秒)")
            _log_info("🔄 回退到高级过渡方法...")
            return self._concat_with_advanced_transitions(video_paths, output_path, quality, transition_duration, "dissolve", motion_compensation, True)

        except Exception as e:
            _log_error(f"形态学过渡拼接失败: {str(e)}")
            # 回退到高级过渡
            return self._concat_with_advanced_transitions(video_paths, output_path, quality, transition_duration, "dissolve", motion_compensation, True)

    def _build_morphing_filter(self, video_paths, transition_duration, motion_compensation):
        """构建形态学过渡滤镜（优化版，更稳定）"""
        try:
            # 获取视频时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        durations.append(float(result.stdout.strip()))
                    else:
                        durations.append(4.0)
                except:
                    durations.append(4.0)

            if len(durations) < 2:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            duration1, duration2 = durations[0], durations[1]
            max_transition = min(duration1, duration2) / 2
            actual_transition = min(transition_duration, max_transition)

            if actual_transition <= 0:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            offset_time = duration1 - actual_transition

            # 简化的形态学过渡滤镜 - 更稳定的实现
            filter_parts = []

            # 运动补偿（如果启用）
            if motion_compensation:
                filter_parts.extend([
                    "[0:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v0smooth]",
                    "[1:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v1smooth]"
                ])
                # 使用高质量dissolve过渡
                filter_parts.append(
                    f"[v0smooth][v1smooth]xfade=transition=dissolve:duration={actual_transition}:offset={offset_time}[output]"
                )
            else:
                # 不使用运动补偿时，使用边缘增强的dissolve
                filter_parts.extend([
                    "[0:v]unsharp=5:5:1.0:5:5:0.0[v0enhanced]",
                    "[1:v]unsharp=5:5:1.0:5:5:0.0[v1enhanced]"
                ])
                filter_parts.append(
                    f"[v0enhanced][v1enhanced]xfade=transition=dissolve:duration={actual_transition}:offset={offset_time}[output]"
                )

            return ";".join(filter_parts)

        except Exception as e:
            _log_error(f"构建形态学过渡滤镜失败: {str(e)}")
            # 回退到简单的dissolve过渡
            offset_time = max(0, durations[0] - transition_duration) if durations else 0
            return f"[0:v][1:v]xfade=transition=dissolve:duration={transition_duration}:offset={offset_time}[output]"

    def _concat_with_optical_flow_transitions(self, video_paths, output_path, quality, transition_duration=0.5):
        """使用光流过渡拼接视频（最高级）"""
        try:
            import subprocess

            if len(video_paths) < 2:
                return self._concat_videos(video_paths, output_path, quality, True, transition_duration, True)

            _log_info(f"🌊 使用光流过渡拼接 {len(video_paths)} 个视频...")

            # 获取视频信息
            video_info = self._analyze_video_properties(video_paths)
            if not video_info:
                _log_error("无法分析视频属性，回退到形态学过渡")
                return self._concat_with_morphing_transitions(video_paths, output_path, quality, transition_duration, True)

            # 现在FFmpeg参数已修复，可以支持各种分辨率的光流过渡
            # 但对于超大分辨率视频，仍然建议使用快速方法以保证用户体验
            if 'target_width' in video_info and 'target_height' in video_info:
                pixels = video_info['target_width'] * video_info['target_height']
                if pixels > 2073600:  # 大于2MP (如1920x1080)，提醒用户但不强制跳过
                    _log_info(f"⚠️ 检测到超大分辨率视频 ({video_info['target_width']}x{video_info['target_height']})，光流过渡可能较慢")
                    _log_info("💡 如需快速处理，建议使用concat_advanced方法")
                    # 不再强制跳过，让用户选择

            # 尝试真正的光流过渡处理
            if len(video_paths) == 2:
                filter_complex = self._build_optical_flow_filter(video_paths, transition_duration)
            else:
                # 多视频光流过渡：使用链式光流处理
                _log_info("🌊 执行多视频光流过渡处理...")
                filter_complex = self._build_optical_flow_multiple_filter(video_paths, transition_duration)

                # 如果多视频光流滤镜构建失败，回退到高级过渡
                if filter_complex is None:
                    _log_info("🔄 多视频光流过渡构建失败，回退到高级过渡方法...")
                    return self._concat_with_advanced_transitions(video_paths, output_path, quality, transition_duration, "radial", False, False)

            inputs = []
            for video_path in video_paths:
                inputs.extend(['-i', video_path])

            quality_params = self._get_quality_params(quality)

            cmd = [
                'ffmpeg'
            ] + inputs + [
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

            _log_info(f"🔧 执行光流过渡命令...")

            # 为真正的光流过渡设置合理的超时时间
            video_info = self._analyze_video_properties(video_paths)

            if video_info and 'target_width' in video_info and 'target_height' in video_info:
                pixels = video_info['target_width'] * video_info['target_height']
                if pixels > 2073600:  # 大于2MP (1920x1080)
                    timeout_seconds = 600  # 10分钟，轻量级光流
                elif pixels > 800000:  # 大于0.8MP (1248x704)
                    timeout_seconds = 480  # 8分钟，标准光流
                else:
                    timeout_seconds = 300  # 5分钟，高质量光流
            else:
                timeout_seconds = 300  # 默认5分钟

            _log_info(f"⏱️ 光流过渡超时设置: {timeout_seconds}秒 (真正光流处理)")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 光流过渡拼接成功")
                return True
            else:
                _log_error(f"❌ 光流过渡拼接失败")
                if result.stderr:
                    _log_error(f"FFmpeg错误: {result.stderr[:300]}...")
                # 回退到形态学过渡
                _log_info("🔄 回退到形态学过渡方法...")
                return self._concat_with_morphing_transitions(video_paths, output_path, quality, transition_duration, True)

        except subprocess.TimeoutExpired:
            _log_error(f"⏰ 光流过渡超时 ({timeout_seconds}秒)")
            _log_info("🔄 回退到形态学过渡方法...")
            return self._concat_with_morphing_transitions(video_paths, output_path, quality, transition_duration, True)

        except Exception as e:
            _log_error(f"光流过渡拼接失败: {str(e)}")
            # 回退到形态学过渡
            return self._concat_with_morphing_transitions(video_paths, output_path, quality, transition_duration, True)

    def _build_optical_flow_filter(self, video_paths, transition_duration):
        """构建光流过渡滤镜（简化版，更稳定）"""
        try:
            # 获取视频时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        durations.append(float(result.stdout.strip()))
                    else:
                        durations.append(4.0)
                except:
                    durations.append(4.0)

            if len(durations) < 2:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            duration1, duration2 = durations[0], durations[1]
            max_transition = min(duration1, duration2) / 2
            actual_transition = min(transition_duration, max_transition)

            if actual_transition <= 0:
                return "[0:v][1:v]concat=n=2:v=1[output]"

            offset_time = duration1 - actual_transition

            # 现在提供真正的光流过渡选项
            # 用户可以选择不同级别的光流处理

            # 获取视频分辨率信息
            pixels = 1248 * 704  # 默认值
            if len(video_paths) >= 2:
                try:
                    # 尝试获取实际分辨率
                    cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_paths[0]]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'x' in result.stdout:
                        width, height = map(int, result.stdout.strip().split('x'))
                        pixels = width * height
                except:
                    pass

            # 根据分辨率选择光流算法复杂度
            if pixels > 2073600:  # 大于2MP (1920x1080)
                _log_info("🌊 使用轻量级光流过渡（适合大分辨率）")
                # 轻量级光流：仅使用基础运动补偿
                filter_parts = [
                    "[0:v]minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir[v0flow]",
                    "[1:v]minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir[v1flow]",
                    f"[v0flow][v1flow]xfade=transition=radial:duration={actual_transition}:offset={offset_time}[output]"
                ]
            elif pixels > 800000:  # 大于0.8MP (1248x704)
                _log_info("🌊 使用标准光流过渡（平衡质量与速度）")
                # 标准光流：中等复杂度
                filter_parts = [
                    "[0:v]minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1[v0flow]",
                    "[1:v]minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1[v1flow]",
                    f"[v0flow][v1flow]xfade=transition=radial:duration={actual_transition}:offset={offset_time}[output]"
                ]
            else:
                _log_info("🌊 使用高质量光流过渡（适合小分辨率）")
                # 高质量光流：完整算法
                filter_parts = [
                    "[0:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff[v0flow]",
                    "[1:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff[v1flow]",
                    f"[v0flow][v1flow]xfade=transition=radial:duration={actual_transition}:offset={offset_time}[output]"
                ]

            return ";".join(filter_parts)

            return ";".join(filter_parts)

        except Exception as e:
            _log_error(f"构建光流过渡滤镜失败: {str(e)}")
            # 回退到高质量的smoothleft过渡
            offset_time = max(0, durations[0] - transition_duration) if durations else 0
            return f"[0:v][1:v]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[output]"

    def _build_optical_flow_multiple_filter(self, video_paths, transition_duration):
        """构建多视频光流过渡滤镜链"""
        try:
            # 获取所有视频时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        duration = float(result.stdout.strip())
                        durations.append(duration)
                    else:
                        _log_error(f"ffprobe失败: {video_path}, 返回码: {result.returncode}")
                        _log_error(f"stderr: {result.stderr}")
                        durations.append(4.0)
                except Exception as e:
                    _log_error(f"ffprobe异常: {video_path}, 错误: {str(e)}")
                    durations.append(4.0)

            if len(durations) < 2:
                return "[0:v]concat=n=1:v=1[output]"

            # 获取视频分辨率信息用于选择光流算法复杂度
            pixels = 1248 * 704  # 默认值
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_paths[0]]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'x' in result.stdout:
                    width, height = map(int, result.stdout.strip().split('x'))
                    pixels = width * height
            except:
                pass

            # 根据分辨率选择光流算法复杂度
            if pixels > 2073600:  # 大于2MP (1920x1080)
                _log_info("🌊 多视频轻量级光流过渡（适合大分辨率）")
                minterpolate_params = "fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"
            elif pixels > 800000:  # 大于0.8MP (1248x704)
                _log_info("🌊 多视频快速光流过渡（优化处理速度）")
                minterpolate_params = "fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"  # 降低fps提高速度
            else:
                _log_info("🌊 多视频标准光流过渡（适合小分辨率）")
                minterpolate_params = "fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"

            filter_parts = []

            # 首先对所有输入视频应用光流处理
            for i in range(len(video_paths)):
                filter_parts.append(f"[{i}:v]minterpolate={minterpolate_params}[v{i}flow]")

            # 使用正确的多视频光流过渡方法：链式处理，但需要正确计算每个过渡的时长
            #
            # 关键理解：xfade的offset是相对于第一个输入的时长，而不是绝对时间
            # 每个xfade输出的长度 = offset + transition_duration

            # 第一个过渡：video1 + video2
            offset_time = durations[0] - transition_duration  # 11.5秒
            filter_parts.append(f"[v0flow][v1flow]xfade=transition=radial:duration={transition_duration}:offset={offset_time}[v01]")
            # v01的长度 = durations[0] + durations[1] - transition_duration = 23.5秒

            # 后续过渡需要重新思考：我们需要将v01和后续视频拼接
            # 但v01已经是23.5秒的完整视频，我们需要在其末尾添加新视频

            current_video_length = durations[0] + durations[1] - transition_duration  # v01的长度

            for i in range(2, len(video_paths)):
                if i == 2:
                    input_label = "v01"
                    output_label = "v02" if i < len(video_paths) - 1 else "output"
                else:
                    input_label = f"v0{i-1}"
                    output_label = f"v0{i}" if i < len(video_paths) - 1 else "output"

                # 对于后续过渡，offset应该是当前视频长度减去过渡时间
                offset_time = current_video_length - transition_duration
                filter_parts.append(f"[{input_label}][v{i}flow]xfade=transition=radial:duration={transition_duration}:offset={offset_time}[{output_label}]")

                # 更新当前视频长度：加上新视频长度，减去过渡时间
                current_video_length += durations[i] - transition_duration

            return ";".join(filter_parts)

        except Exception as e:
            _log_error(f"构建多视频光流过渡滤镜失败: {str(e)}")
            # 回退到高级过渡方法
            _log_info("🔄 光流过渡失败，回退到高级过渡方法...")
            return None  # 返回None表示需要回退

    def _concat_morphing_multiple(self, video_paths, output_path, quality, transition_duration, motion_compensation):
        """多视频形态学过渡拼接 - 保留旧方法作为备用"""
        _log_info("🔄 使用旧的递归形态学过渡方法（备用）")
        return self._concat_morphing_multiple_chain(video_paths, output_path, quality, transition_duration, motion_compensation)

    def _concat_morphing_multiple_chain(self, video_paths, output_path, quality, transition_duration, motion_compensation):
        """使用一次性滤镜链的形态学过渡拼接 - 修复时长计算"""
        try:
            import subprocess

            # 获取所有视频的时长
            durations = []
            for video_path in video_paths:
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        durations.append(float(result.stdout.strip()))
                    else:
                        durations.append(4.0)  # 默认4秒
                except:
                    durations.append(4.0)

            if len(durations) < 2:
                return False

            # 计算过渡参数
            min_duration = min(durations)
            max_transition = min_duration / 2
            actual_transition = min(transition_duration, max_transition)

            if actual_transition <= 0:
                # 无过渡，使用简单concat
                return self._simple_concat_multiple(video_paths, output_path)

            # 构建多视频形态学滤镜链
            filter_complex = self._build_multiple_morphing_chain(video_paths, durations, actual_transition, motion_compensation)

            if not filter_complex:
                _log_error("构建多视频形态学滤镜链失败")
                return False

            # 执行FFmpeg命令
            cmd = ['ffmpeg']

            # 添加输入文件
            for video_path in video_paths:
                cmd.extend(['-i', video_path])

            # 添加滤镜和输出参数
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[output]',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',
                '-crf', '23',
                '-y',
                output_path
            ])

            _log_info(f"🔧 执行多视频形态学过渡命令...")

            # 根据视频分辨率调整超时时间
            video_info = self._analyze_video_properties(video_paths)
            base_timeout = 20  # 基础20秒超时

            if video_info and 'target_width' in video_info and 'target_height' in video_info:
                pixels = video_info['target_width'] * video_info['target_height']
                if pixels > 1000000:  # 大于1MP (如1248x704)
                    timeout_seconds = 45  # 最多45秒
                elif pixels > 500000:  # 大于0.5MP
                    timeout_seconds = 30  # 30秒
                else:
                    timeout_seconds = 20  # 20秒
            else:
                timeout_seconds = 20

            _log_info(f"⏱️ 形态学过渡超时设置: {timeout_seconds}秒 (快速处理策略)")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info("✅ 多视频形态学过渡拼接成功")
                return True
            else:
                _log_error(f"多视频形态学过渡拼接失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            _log_error(f"⏰ 形态学过渡超时 ({timeout_seconds}秒)")
            return False
        except Exception as e:
            _log_error(f"多视频形态学过渡拼接异常: {str(e)}")
            return False

    def _build_multiple_morphing_chain(self, video_paths, durations, transition_duration, motion_compensation):
        """构建多视频形态学滤镜链"""
        try:
            if len(video_paths) < 2:
                return None

            if len(video_paths) == 2:
                # 两个视频的简单情况
                offset_time = durations[0] - transition_duration

                # 简化的形态学过渡滤镜
                if motion_compensation:
                    # 带运动补偿的形态学过渡
                    filter_parts = [
                        "[0:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v0smooth]",
                        "[1:v]minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc[v1smooth]",
                        f"[v0smooth][v1smooth]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[output]"
                    ]
                else:
                    # 简单的形态学过渡
                    filter_parts = [
                        "[0:v]edgedetect=low=0.1:high=0.4[v0edge]",
                        "[1:v]edgedetect=low=0.1:high=0.4[v1edge]",
                        f"[v0edge][v1edge]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[output]"
                    ]

                return ";".join(filter_parts)

            # 多个视频的复杂情况 - 使用简化的过渡效果
            filter_parts = []
            current_offset = 0

            # 第一个过渡
            offset_time = durations[0] - transition_duration
            if motion_compensation:
                filter_parts.extend([
                    "[0:v]minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc[v0smooth]",
                    "[1:v]minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc[v1smooth]",
                    f"[v0smooth][v1smooth]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[v01]"
                ])
            else:
                filter_parts.append(f"[0:v][1:v]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[v01]")

            current_offset = durations[0] + durations[1] - transition_duration

            # 后续过渡
            for i in range(2, len(video_paths)):
                input_label = f"v0{i-1}" if i == 2 else f"v0{i-1}"
                output_label = f"v0{i}" if i < len(video_paths) - 1 else "output"

                # 计算这个过渡的偏移时间
                offset_time = current_offset - transition_duration

                if motion_compensation and i == 2:  # 只对第二个过渡使用运动补偿，避免过于复杂
                    filter_parts.extend([
                        f"[{i}:v]minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc[v{i}smooth]",
                        f"[{input_label}][v{i}smooth]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[{output_label}]"
                    ])
                else:
                    filter_parts.append(f"[{input_label}][{i}:v]xfade=transition=smoothleft:duration={transition_duration}:offset={offset_time}[{output_label}]")

                current_offset += durations[i] - transition_duration

            return ";".join(filter_parts)

        except Exception as e:
            _log_error(f"构建多视频形态学滤镜链失败: {str(e)}")
            return None

    def _concat_optical_flow_multiple(self, video_paths, output_path, quality, transition_duration):
        """多视频光流过渡拼接"""
        try:
            import tempfile

            temp_dir = tempfile.mkdtemp()
            intermediate_files = []

            try:
                current_video = video_paths[0]

                for i in range(1, len(video_paths)):
                    next_video = video_paths[i]
                    temp_output = os.path.join(temp_dir, f"flow_intermediate_{i}.mp4")

                    success = self._concat_with_optical_flow_transitions(
                        [current_video, next_video],
                        temp_output,
                        quality,
                        transition_duration
                    )

                    if not success:
                        _log_error(f"光流过渡中间步骤 {i} 失败")
                        return False

                    intermediate_files.append(temp_output)
                    current_video = temp_output

                # 复制最终结果
                if intermediate_files:
                    final_temp = intermediate_files[-1]
                    if os.path.exists(final_temp):
                        import shutil
                        shutil.copy2(final_temp, output_path)
                        return os.path.exists(output_path)

                return False

            finally:
                # 清理临时文件
                for temp_file in intermediate_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass

        except Exception as e:
            _log_error(f"多视频光流过渡拼接失败: {str(e)}")
            return False

    def _hstack_videos(self, video_paths, output_path, quality, scale_videos):
        """水平拼接视频（并排显示）"""
        try:
            import subprocess

            _log_info("↔️ 使用hstack方法拼接视频...")

            if len(video_paths) > 8:
                _log_error("hstack方法最多支持8个视频")
                return False

            # 构建FFmpeg filter_complex
            inputs = []
            for i, video_path in enumerate(video_paths):
                inputs.extend(['-i', video_path])

            # 构建缩放和拼接滤镜
            if scale_videos:
                # 先缩放到统一尺寸，再水平拼接
                scale_filters = []
                for i in range(len(video_paths)):
                    scale_filters.append(f"[{i}:v]scale=640:480[v{i}]")

                hstack_filter = "[" + "][".join([f"v{i}" for i in range(len(video_paths))]) + "]hstack=inputs=" + str(len(video_paths)) + "[outv]"
                filter_complex = ";".join(scale_filters) + ";" + hstack_filter
            else:
                # 直接拼接
                input_labels = "[" + "][".join([f"{i}:v" for i in range(len(video_paths))]) + "]"
                filter_complex = input_labels + "hstack=inputs=" + str(len(video_paths)) + "[outv]"

            quality_params = self._get_quality_params(quality)
            cmd = [
                'ffmpeg'
            ] + inputs + [
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '0:a?',  # 使用第一个视频的音频
            ] + quality_params + [
                '-y',
                output_path
            ]

            _log_info(f"🔧 执行FFmpeg命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            _log_error(f"hstack拼接失败: {str(e)}")
            return False

    def _vstack_videos(self, video_paths, output_path, quality, scale_videos):
        """垂直拼接视频（上下显示）"""
        try:
            import subprocess

            _log_info("↕️ 使用vstack方法拼接视频...")

            if len(video_paths) > 8:
                _log_error("vstack方法最多支持8个视频")
                return False

            # 构建FFmpeg filter_complex
            inputs = []
            for i, video_path in enumerate(video_paths):
                inputs.extend(['-i', video_path])

            # 构建缩放和拼接滤镜
            if scale_videos:
                # 先缩放到统一尺寸，再垂直拼接
                scale_filters = []
                for i in range(len(video_paths)):
                    scale_filters.append(f"[{i}:v]scale=640:480[v{i}]")

                vstack_filter = "[" + "][".join([f"v{i}" for i in range(len(video_paths))]) + "]vstack=inputs=" + str(len(video_paths)) + "[outv]"
                filter_complex = ";".join(scale_filters) + ";" + vstack_filter
            else:
                # 直接拼接
                input_labels = "[" + "][".join([f"{i}:v" for i in range(len(video_paths))]) + "]"
                filter_complex = input_labels + "vstack=inputs=" + str(len(video_paths)) + "[outv]"

            quality_params = self._get_quality_params(quality)
            cmd = [
                'ffmpeg'
            ] + inputs + [
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '0:a?',  # 使用第一个视频的音频
            ] + quality_params + [
                '-y',
                output_path
            ]

            _log_info(f"🔧 执行FFmpeg命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            _log_error(f"vstack拼接失败: {str(e)}")
            return False

    def _grid_videos(self, video_paths, output_path, quality, grid_type, scale_videos):
        """网格拼接视频（2x2、2x3或2x4布局）"""
        try:
            import subprocess

            _log_info(f"🔲 使用{grid_type}网格方法拼接视频...")

            if grid_type == "2x2":
                max_videos = 4
            elif grid_type == "2x3":
                max_videos = 6
            elif grid_type == "2x4":
                max_videos = 8
            else:
                max_videos = 4

            if len(video_paths) > max_videos:
                _log_error(f"{grid_type}网格最多支持{max_videos}个视频")
                return False

            # 构建FFmpeg filter_complex
            inputs = []
            for i, video_path in enumerate(video_paths):
                inputs.extend(['-i', video_path])

            # 为不足的位置创建黑色视频
            while len(video_paths) < max_videos:
                video_paths.append(None)

            # 构建网格滤镜
            if scale_videos:
                # 缩放所有视频到统一尺寸
                scale_filters = []
                for i in range(len([v for v in video_paths if v is not None])):
                    scale_filters.append(f"[{i}:v]scale=320:240[v{i}]")

                # 为空位置创建黑色视频
                black_filters = []
                actual_videos = len([v for v in video_paths if v is not None])
                for i in range(actual_videos, max_videos):
                    black_filters.append(f"color=black:320x240:d=1[v{i}]")

                if grid_type == "2x2":
                    # 2x2网格布局
                    grid_filter = "[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[outv]"
                elif grid_type == "2x3":
                    # 2x3网格布局
                    grid_filter = "[v0][v1]hstack[top];[v2][v3]hstack[middle];[v4][v5]hstack[bottom];[top][middle]vstack[temp];[temp][bottom]vstack[outv]"
                else:  # 2x4
                    # 2x4网格布局
                    grid_filter = "[v0][v1]hstack[row1];[v2][v3]hstack[row2];[v4][v5]hstack[row3];[v6][v7]hstack[row4];[row1][row2]vstack[temp1];[row3][row4]vstack[temp2];[temp1][temp2]vstack[outv]"

                all_filters = scale_filters + black_filters + [grid_filter]
                filter_complex = ";".join(all_filters)
            else:
                # 不缩放，直接网格拼接（可能会有尺寸不匹配问题）
                black_filters = []
                actual_videos = len([v for v in video_paths if v is not None])
                for i in range(actual_videos, max_videos):
                    black_filters.append(f"color=black:640x480:d=1[v{i}]")

                if grid_type == "2x2":
                    grid_filter = f"[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[outv]"
                elif grid_type == "2x3":
                    grid_filter = f"[0:v][1:v]hstack[top];[2:v][3:v]hstack[middle];[4:v][5:v]hstack[bottom];[top][middle]vstack[temp];[temp][bottom]vstack[outv]"
                else:  # 2x4
                    grid_filter = f"[0:v][1:v]hstack[row1];[2:v][3:v]hstack[row2];[4:v][5:v]hstack[row3];[6:v][7:v]hstack[row4];[row1][row2]vstack[temp1];[row3][row4]vstack[temp2];[temp1][temp2]vstack[outv]"

                filter_complex = ";".join(black_filters + [grid_filter])

            quality_params = self._get_quality_params(quality)
            cmd = [
                'ffmpeg'
            ] + inputs + [
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '0:a?',  # 使用第一个视频的音频
            ] + quality_params + [
                '-y',
                output_path
            ]

            _log_info(f"🔧 执行FFmpeg命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            _log_error(f"grid拼接失败: {str(e)}")
            return False

    def _create_error_result(self, error_msg):
        """创建错误结果"""
        try:
            blank_video = create_blank_video_object()
            blank_video_path = getattr(blank_video, 'file_path', '') if blank_video else ''
            afvideo = create_video_path_wrapper(blank_video_path) if blank_video_path else create_blank_video_object()
            return (blank_video, f"❌ {error_msg}", afvideo)
        except:
            return (None, f"❌ {error_msg}", None)


class GetLastFrameNode:
    """提取任意视频尾帧的独立节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "output_filename": ("STRING", {"default": ""}),
                "image_quality": (["high", "medium", "low"], {"default": "high"}),
                "offset_from_last": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("last_frame_image", "frame_path")
    FUNCTION = "extract_last_frame"
    CATEGORY = "Ken-Chen/Video_Utilities"

    def __init__(self):
        self.timeout = 60  # 1分钟超时

    def extract_last_frame(self, video, output_filename="", image_quality="high", offset_from_last=0):
        """
        提取视频的最后一帧或从尾部起第N帧（0表示尾帧）

        Args:
            video: ComfyUI VIDEO对象
            output_filename: 输出文件名（可选）
            image_quality: 图像质量设置
            offset_from_last: 从尾部起的偏移帧数（0=尾帧，1=倒数第2帧...）

        Returns:
            tuple: (图像张量, 图像文件路径)
        """
        try:
            _log_info("🎬 开始提取视频尾帧...")

            # 获取视频文件路径 - 使用改进的提取方法
            video_path = self._extract_video_path(video)

            if not video_path:
                error_msg = f"无法获取有效的视频文件路径: {video_path}"
                _log_error(error_msg)
                _log_error(f"视频对象详情: type={type(video)}, repr={repr(video)}")
                # 返回空白图像和错误信息
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            if not os.path.exists(video_path):
                error_msg = f"视频文件不存在: {video_path}"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            _log_info(f"📹 视频文件路径: {video_path}")

            # 生成输出文件路径
            if not output_filename:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                if offset_from_last and offset_from_last > 0:
                    output_filename = f"{video_name}_last_minus_{offset_from_last}.jpg"
                else:
                    output_filename = f"{video_name}_last_frame.jpg"

            # 确保输出文件名有正确的扩展名
            if not output_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                output_filename += '.jpg'

            # 使用临时目录
            import tempfile
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{int(time.time())}_{output_filename}")

            # 设置图像质量参数
            quality_settings = {
                "high": ["-q:v", "2"],      # 高质量
                "medium": ["-q:v", "5"],    # 中等质量
                "low": ["-q:v", "8"]        # 低质量
            }
            quality_params = quality_settings.get(image_quality, quality_settings["high"])

            # 提取指定帧（支持从尾部偏移）
            frame_path = None
            try:
                # 优先尝试使用帧序号方式
                total_frames = self._get_total_frames(video_path)
                if isinstance(total_frames, int) and total_frames > 0:
                    target_index = max(total_frames - 1 - max(0, int(offset_from_last or 0)), 0)
                    frame_path = self._extract_frame_by_index(video_path, output_path, quality_params, target_index)
                else:
                    frame_path = None
            except Exception:
                frame_path = None

            # 退回策略：当不能精确按帧索引时，使用时间估算或尾帧方式
            if not frame_path:
                if offset_from_last and offset_from_last > 0:
                    # 使用时长和fps估算定位
                    duration, fps = self._get_duration_and_fps(video_path)
                    if duration and fps:
                        seek_time = max(0.0, float(duration) - (float(offset_from_last) + 1.0) / float(fps))
                        frame_path = self._extract_frame_by_time(video_path, output_path, quality_params, seek_time)
                # 最后兜底为尾帧
                if not frame_path:
                    frame_path = self._extract_frame_with_ffmpeg(video_path, output_path, quality_params)

            if not frame_path:
                error_msg = "尾帧提取失败"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            # 将图像转换为ComfyUI张量格式
            image_tensor = self._load_image_as_tensor(frame_path)

            if image_tensor is None:
                error_msg = "图像加载失败"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            _log_info(f"✅ 尾帧提取成功: {frame_path}")
            return (image_tensor, frame_path)

        except Exception as e:
            error_msg = f"提取视频尾帧失败: {str(e)}"
            _log_error(error_msg)
            blank_image = self._create_blank_image()
            return (blank_image, f"❌ {error_msg}")

    def _extract_frame_by_index(self, video_path, output_path, quality_params, frame_index):
        """使用帧索引提取指定帧（0-based）"""
        try:
            import subprocess
            # 使用select=eq(n,frame_index) 精确选帧
            vf_expr = f"select='eq(n,{int(frame_index)})'"
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', vf_expr,
                '-vsync', 'vfr',
                '-frames:v', '1',
            ] + quality_params + [
                '-y',
                output_path
            ]
            _log_info(f"🔧 按索引提取帧: index={frame_index}, cmd={' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception as e:
            _log_error(f"按索引提取失败: {str(e)}")
            return None

    def _extract_frame_by_time(self, video_path, output_path, quality_params, seek_time):
        """按时间定位提取单帧（seek_time为秒）"""
        try:
            import subprocess
            cmd = [
                'ffmpeg',
                '-ss', f"{seek_time}",
                '-i', video_path,
                '-frames:v', '1',
            ] + quality_params + [
                '-y',
                output_path
            ]
            _log_info(f"🔧 按时间提取帧: t={seek_time:.3f}s, cmd={' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception as e:
            _log_error(f"按时间提取失败: {str(e)}")
            return None

    def _get_total_frames(self, video_path):
        """使用ffprobe尽可能获取总帧数，失败返回None"""
        try:
            import subprocess, json
            # 方案1：count_frames
            cmd1 = [
                'ffprobe',
                '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                video_path
            ]
            res1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=20)
            if res1.returncode == 0:
                val = res1.stdout.strip()
                if val.isdigit():
                    return int(val)

            # 方案2：读取nb_frames
            cmd2 = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                video_path
            ]
            res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=20)
            if res2.returncode == 0:
                val = res2.stdout.strip()
                if val.isdigit():
                    return int(val)
        except Exception as e:
            _log_warning(f"无法获取总帧数: {str(e)}")
        return None

    def _get_duration_and_fps(self, video_path):
        """返回(duration_seconds, fps) 或 (None, None)"""
        try:
            import subprocess, json
            # 获取时长
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate:format=duration',
                '-of', 'json',
                video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                duration = None
                fps = None
                if 'format' in data and 'duration' in data['format']:
                    try:
                        duration = float(data['format']['duration'])
                    except:
                        duration = None
                if 'streams' in data and data['streams']:
                    afr = data['streams'][0].get('avg_frame_rate')
                    if afr and afr != '0/0':
                        try:
                            num, den = afr.split('/')
                            num = float(num)
                            den = float(den) if float(den) != 0 else 1.0
                            fps = num / den if den else None
                        except:
                            fps = None
                return duration, fps
        except Exception as e:
            _log_warning(f"无法获取时长与FPS: {str(e)}")
        return None, None

    def _extract_frame_with_ffmpeg(self, video_path, output_path, quality_params):
        """使用FFmpeg提取尾帧"""
        try:
            import subprocess

            # 方法1：使用select=eof过滤器
            cmd1 = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'select=eof',
                '-vsync', 'vfr',
                '-frames:v', '1',
            ] + quality_params + [
                '-y',
                output_path
            ]

            _log_info(f"🔧 执行FFmpeg命令: {' '.join(cmd1)}")

            result = subprocess.run(
                cmd1,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path

            # 方法2：备用时长计算方法
            _log_info("🔄 尝试备用方法...")

            # 获取视频时长
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

            if duration_result.returncode == 0:
                try:
                    duration = float(duration_result.stdout.strip())
                    seek_time = max(0, duration - 0.1)

                    cmd2 = [
                        'ffmpeg',
                        '-ss', str(seek_time),
                        '-i', video_path,
                        '-frames:v', '1',
                    ] + quality_params + [
                        '-y',
                        output_path
                    ]

                    result = subprocess.run(
                        cmd2,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout
                    )

                    if result.returncode == 0 and os.path.exists(output_path):
                        return output_path
                except:
                    pass

            return None

        except Exception as e:
            _log_error(f"FFmpeg提取失败: {str(e)}")
            return None

    def _load_image_as_tensor(self, image_path):
        """将图像文件加载为ComfyUI张量格式"""
        try:
            from PIL import Image
            import numpy as np
            import torch

            # 使用PIL加载图像
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 转换为numpy数组
                img_array = np.array(img).astype(np.float32) / 255.0

                # 添加batch维度 [H, W, C] -> [1, H, W, C]
                img_array = np.expand_dims(img_array, axis=0)

                # 转换为torch张量（ComfyUI期望的格式）
                img_tensor = torch.from_numpy(img_array)

                _log_info(f"✅ 图像张量格式: {img_tensor.shape}, dtype: {img_tensor.dtype}")
                return img_tensor

        except Exception as e:
            _log_error(f"图像加载失败: {str(e)}")
            return None

    def _extract_video_path(self, video):
        """从VIDEO对象提取文件路径"""
        _log_info(f"🔍 尝试从VIDEO对象提取路径: {type(video)}")

        # 如果是字符串，直接返回
        if isinstance(video, str):
            _log_info(f"✅ 直接字符串路径: {video}")
            return video

        # 尝试常见的文件路径属性
        path_attributes = [
            'file_path',    # 我们自己的VideoFromFile对象
            'filename',     # 一些节点使用这个
            'file',         # 向后兼容
            'path',         # 通用路径属性
            'filepath',     # 文件路径
            'video_path',   # 视频路径
            'source',       # 源文件
            'url',          # URL路径
            'video_file',   # 视频文件
            'file_name',    # 文件名
        ]

        for attr in path_attributes:
            if hasattr(video, attr):
                value = getattr(video, attr)
                if value and isinstance(value, str):
                    _log_info(f"✅ 从属性 {attr} 获取路径: {value}")
                    return value
                elif value:
                    _log_info(f"⚠️ 属性 {attr} 存在但不是字符串: {type(value)} = {value}")

        # 如果是字典类型，尝试从字典中获取路径
        if isinstance(video, dict):
            for key in ['file_path', 'filename', 'path', 'url', 'source']:
                if key in video and isinstance(video[key], str):
                    _log_info(f"✅ 从字典键 {key} 获取路径: {video[key]}")
                    return video[key]

        # 如果有__dict__属性，打印所有属性用于调试
        if hasattr(video, '__dict__'):
            _log_info(f"🔍 VIDEO对象属性: {list(video.__dict__.keys())}")
            for key, value in video.__dict__.items():
                if isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower() or 'url' in key.lower()):
                    _log_info(f"✅ 从__dict__属性 {key} 获取路径: {value}")
                    return value

        # 最后尝试：如果对象可以转换为字符串且看起来像路径
        try:
            str_repr = str(video)
            if str_repr and ('/' in str_repr or '\\' in str_repr or str_repr.endswith('.mp4')):
                _log_info(f"✅ 从字符串表示获取路径: {str_repr}")
                return str_repr
        except:
            pass

        _log_error(f"❌ 无法从VIDEO对象提取路径，对象类型: {type(video)}")
        return None

    def _create_blank_image(self):
        """创建空白图像张量"""
        try:
            import numpy as np
            import torch
            # 创建512x512的黑色图像
            blank_array = np.zeros((1, 512, 512, 3), dtype=np.float32)
            # 转换为torch张量（ComfyUI期望的格式）
            blank_tensor = torch.from_numpy(blank_array)
            return blank_tensor
        except:
            return None


class GetFirstFrameNode:
    """提取任意视频首帧的独立节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "output_filename": ("STRING", {"default": ""}),
                "image_quality": (["high", "medium", "low"], {"default": "high"}),
                "offset_from_first": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "从首帧开始的偏移帧数（0=首帧，1=第2帧，2=第3帧...）"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("first_frame_image", "frame_path")
    FUNCTION = "extract_first_frame"
    CATEGORY = "Ken-Chen/Video_Utilities"

    def __init__(self):
        self.timeout = 60  # 1分钟超时

    def extract_first_frame(self, video, output_filename="", image_quality="high", offset_from_first=0):
        """
        提取视频的第一帧或从首帧起第N帧（0表示首帧）

        Args:
            video: ComfyUI VIDEO对象
            output_filename: 输出文件名（可选）
            image_quality: 图像质量设置
            offset_from_first: 从首帧起的偏移帧数（0=首帧，1=第2帧，2=第3帧...）

        Returns:
            tuple: (图像张量, 图像文件路径)
        """
        try:
            _log_info("🎬 开始提取视频首帧...")

            # 获取视频文件路径 - 使用改进的提取方法
            video_path = self._extract_video_path(video)

            if not video_path:
                error_msg = f"无法获取有效的视频文件路径: {video_path}"
                _log_error(error_msg)
                _log_error(f"视频对象详情: type={type(video)}, repr={repr(video)}")
                # 返回空白图像和错误信息
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            if not os.path.exists(video_path):
                error_msg = f"视频文件不存在: {video_path}"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            _log_info(f"📹 视频文件路径: {video_path}")

            # 生成输出文件路径
            if not output_filename:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                if offset_from_first and offset_from_first > 0:
                    output_filename = f"{video_name}_frame_{offset_from_first}.jpg"
                else:
                    output_filename = f"{video_name}_first_frame.jpg"

            # 确保输出文件名有正确的扩展名
            if not output_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                output_filename += '.jpg'

            # 使用临时目录
            import tempfile
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{int(time.time())}_{output_filename}")

            # 设置图像质量参数
            quality_settings = {
                "high": ["-q:v", "2"],      # 高质量
                "medium": ["-q:v", "5"],    # 中等质量
                "low": ["-q:v", "8"]        # 低质量
            }
            quality_params = quality_settings.get(image_quality, quality_settings["high"])

            # 提取指定帧（从首帧开始偏移）
            frame_index = max(0, int(offset_from_first or 0))
            frame_path = self._extract_frame_by_index(video_path, output_path, quality_params, frame_index)

            # 如果按索引提取失败，尝试按时间提取
            if not frame_path and offset_from_first and offset_from_first > 0:
                # 使用fps估算时间定位
                _, fps = self._get_duration_and_fps(video_path)
                if fps:
                    seek_time = float(offset_from_first) / float(fps)
                    frame_path = self._extract_frame_by_time(video_path, output_path, quality_params, seek_time)

            # 最后兜底：提取首帧（时间0）
            if not frame_path:
                frame_path = self._extract_frame_by_time(video_path, output_path, quality_params, 0.0)

            if not frame_path:
                error_msg = "首帧提取失败"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            # 将图像转换为ComfyUI张量格式
            image_tensor = self._load_image_as_tensor(frame_path)

            if image_tensor is None:
                error_msg = "图像加载失败"
                _log_error(error_msg)
                blank_image = self._create_blank_image()
                return (blank_image, f"❌ {error_msg}")

            _log_info(f"✅ 首帧提取成功: {frame_path}")
            return (image_tensor, frame_path)

        except Exception as e:
            error_msg = f"提取视频首帧失败: {str(e)}"
            _log_error(error_msg)
            blank_image = self._create_blank_image()
            return (blank_image, f"❌ {error_msg}")

    def _extract_frame_by_index(self, video_path, output_path, quality_params, frame_index):
        """使用帧索引提取指定帧（0-based）"""
        try:
            import subprocess
            # 使用select=eq(n,frame_index) 精确选帧
            vf_expr = f"select='eq(n,{int(frame_index)})'"
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', vf_expr,
                '-vsync', 'vfr',
                '-frames:v', '1',
            ] + quality_params + [
                '-y',
                output_path
            ]
            _log_info(f"🔧 按索引提取帧: index={frame_index}, cmd={' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception as e:
            _log_error(f"按索引提取失败: {str(e)}")
            return None

    def _extract_frame_by_time(self, video_path, output_path, quality_params, seek_time):
        """按时间定位提取单帧（seek_time为秒）"""
        try:
            import subprocess
            cmd = [
                'ffmpeg',
                '-ss', f"{seek_time}",
                '-i', video_path,
                '-frames:v', '1',
            ] + quality_params + [
                '-y',
                output_path
            ]
            _log_info(f"🔧 按时间提取帧: t={seek_time:.3f}s, cmd={' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception as e:
            _log_error(f"按时间提取失败: {str(e)}")
            return None

    def _get_duration_and_fps(self, video_path):
        """返回(duration_seconds, fps) 或 (None, None)"""
        try:
            import subprocess, json
            # 获取时长和帧率
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate:format=duration',
                '-of', 'json',
                video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                duration = None
                fps = None
                if 'format' in data and 'duration' in data['format']:
                    try:
                        duration = float(data['format']['duration'])
                    except:
                        duration = None
                if 'streams' in data and data['streams']:
                    afr = data['streams'][0].get('avg_frame_rate')
                    if afr and afr != '0/0':
                        try:
                            num, den = afr.split('/')
                            num = float(num)
                            den = float(den) if float(den) != 0 else 1.0
                            fps = num / den if den else None
                        except:
                            fps = None
                return duration, fps
        except Exception as e:
            _log_warning(f"无法获取时长与FPS: {str(e)}")
        return None, None

    def _load_image_as_tensor(self, image_path):
        """将图像文件加载为ComfyUI张量格式"""
        try:
            from PIL import Image
            import numpy as np
            import torch

            # 使用PIL加载图像
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 转换为numpy数组
                img_array = np.array(img).astype(np.float32) / 255.0

                # 添加batch维度 [H, W, C] -> [1, H, W, C]
                img_array = np.expand_dims(img_array, axis=0)

                # 转换为torch张量（ComfyUI期望的格式）
                img_tensor = torch.from_numpy(img_array)

                _log_info(f"✅ 图像张量格式: {img_tensor.shape}, dtype: {img_tensor.dtype}")
                return img_tensor

        except Exception as e:
            _log_error(f"图像加载失败: {str(e)}")
            return None

    def _extract_video_path(self, video):
        """从VIDEO对象提取文件路径"""
        _log_info(f"🔍 尝试从VIDEO对象提取路径: {type(video)}")

        # 如果是字符串，直接返回
        if isinstance(video, str):
            _log_info(f"✅ 直接字符串路径: {video}")
            return video

        # 尝试常见的文件路径属性
        path_attributes = [
            'file_path',    # 我们自己的VideoFromFile对象
            'filename',     # 一些节点使用这个
            'file',         # 向后兼容
            'path',         # 通用路径属性
            'filepath',     # 文件路径
            'video_path',   # 视频路径
            'source',       # 源文件
            'url',          # URL路径
            'video_file',   # 视频文件
            'file_name',    # 文件名
        ]

        for attr in path_attributes:
            if hasattr(video, attr):
                value = getattr(video, attr)
                if value and isinstance(value, str):
                    _log_info(f"✅ 从属性 {attr} 获取路径: {value}")
                    return value
                elif value:
                    _log_info(f"⚠️ 属性 {attr} 存在但不是字符串: {type(value)} = {value}")

        # 如果是字典类型，尝试从字典中获取路径
        if isinstance(video, dict):
            for key in ['file_path', 'filename', 'path', 'url', 'source']:
                if key in video and isinstance(video[key], str):
                    _log_info(f"✅ 从字典键 {key} 获取路径: {video[key]}")
                    return video[key]

        # 如果有__dict__属性，打印所有属性用于调试
        if hasattr(video, '__dict__'):
            _log_info(f"🔍 VIDEO对象属性: {list(video.__dict__.keys())}")
            for key, value in video.__dict__.items():
                if isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower() or 'url' in key.lower()):
                    _log_info(f"✅ 从__dict__属性 {key} 获取路径: {value}")
                    return value

        # 最后尝试：如果对象可以转换为字符串且看起来像路径
        try:
            str_repr = str(video)
            if str_repr and ('/' in str_repr or '\\' in str_repr or str_repr.endswith('.mp4')):
                _log_info(f"✅ 从字符串表示获取路径: {str_repr}")
                return str_repr
        except:
            pass

        _log_error(f"❌ 无法从VIDEO对象提取路径，对象类型: {type(video)}")
        return None

    def _create_blank_image(self):
        """创建空白图像张量"""
        try:
            import numpy as np
            import torch
            # 创建512x512的黑色图像
            blank_array = np.zeros((1, 512, 512, 3), dtype=np.float32)
            # 转换为torch张量（ComfyUI期望的格式）
            blank_tensor = torch.from_numpy(blank_array)
            return blank_tensor
        except:
            return None




# 与UtilNodes-ComfyUI完全一致的节点：VideoUtilitiesGetVHSFilePath
class VideoUtilitiesGetVHSFilePath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES",),
                "sleep": ("INT", {"default": 3, "min": 0, "max": 60, "step": 1}),
            }
        }

    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Get video file path from Video Combine node output"

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("AFVIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "get_video_path"

    def get_video_path(self, filenames, sleep):
        import time, os, shutil

        if sleep > 0:
            time.sleep(sleep)

        # 使用ComfyUI的输出目录
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = tempfile.gettempdir()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 只处理元组或列表且第二项为列表的情况
        if isinstance(filenames, (tuple, list)) and len(filenames) >= 2 and isinstance(filenames[1], list):
            file_paths = filenames[1]
            # 逆序查找最后一个存在的视频文件
            for path in reversed(file_paths):
                if isinstance(path, str) and path.lower().endswith((".mp4", ".webm", ".mkv", ".avi")) and os.path.exists(path):
                    # 拷贝到output目录
                    new_path = os.path.join(output_dir, os.path.basename(path))
                    try:
                        shutil.copy2(path, new_path)
                    except Exception:
                        # 拷贝失败就返回原路径
                        return (path,)
                    return (new_path,)
            # 如果没有找到存在的视频文件，也可以只返回最后一个视频文件名（不判断存在性）
            for path in reversed(file_paths):
                if isinstance(path, str) and path.lower().endswith((".mp4", ".webm", ".mkv", ".avi")):
                    return (path,)
        # 兜底：返回空字符串
        return ("",)

# 将视频转为GIF的节点
class VideoToGIFNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 60, "step": 1}),
                "max_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "max_size_mb": ("INT", {"default": 8, "min": 1, "max": 200, "step": 1}),
                "colors": ("INT", {"default": 128, "min": 2, "max": 256, "step": 1}),
                "dither": (["floyd_steinberg", "bayer", "none"], {"default": "floyd_steinberg"}),
                "loop": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "keep_aspect": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Convert a video file to GIF with controllable quality and size"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("GIF_PATH", "STATUS")

    OUTPUT_NODE = False

    FUNCTION = "convert_to_gif"

    def _extract_video_path(self, video):
        try:
            if isinstance(video, str):
                return video
            for attr in ["file_path", "filename", "file", "path", "filepath", "video_path"]:
                if hasattr(video, attr):
                    val = getattr(video, attr)
                    if isinstance(val, str) and val:
                        return val
            if isinstance(video, dict):
                for k in ["file_path", "filename", "path", "url", "source"]:
                    if k in video and isinstance(video[k], str):
                        return video[k]
            s = str(video)
            if s and ("/" in s or "\\" in s) and s.lower().endswith((".mp4", ".webm", ".mkv", ".avi")):
                return s
        except:
            pass
        return None

    def _ensure_output_dir(self):
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _run_ffmpeg_gif(self, src, dst, fps, width, colors, dither, loop):
        palette = dst + ".palette.png"
        vf_scale = f"scale={width}:-1:flags=lanczos"
        palettegen = [
            "-vf", f"fps={fps},{vf_scale},palettegen=max_colors={colors}:stats_mode=full"
        ]
        paletteuse = [
            "-lavfi", f"fps={fps},{vf_scale} [x]; [x][1:v] paletteuse=dither={dither}:diff_mode=rectangle"
        ]
        # 生成调色板
        cmd1 = [
            "ffmpeg", "-y", "-i", src,
        ] + palettegen + [palette]
        # 使用调色板生成gif
        cmd2 = [
            "ffmpeg", "-y", "-i", src, "-i", palette,
            "-loop", str(loop),
        ] + paletteuse + [dst]

        r1 = subprocess.run(cmd1, capture_output=True)
        if r1.returncode != 0:
            return False, f"palettegen failed: {r1.stderr.decode(errors='ignore')[:200]}"
        r2 = subprocess.run(cmd2, capture_output=True)
        try:
            os.unlink(palette)
        except:
            pass
        if r2.returncode != 0:
            return False, f"gif encode failed: {r2.stderr.decode(errors='ignore')[:200]}"
        return True, "ok"

    def convert_to_gif(self, video, fps, max_width, max_size_mb, colors, dither, loop, keep_aspect=True):
        try:
            _log_info("[Video_To_GIF] start")
            src = self._extract_video_path(video)
            if not src or not os.path.exists(src):
                _log_error(f"[Video_To_GIF] source not found: {src}")
                return {"ui": {}, "result": ("", f"Video not found: {src}")}

            out_dir = self._ensure_output_dir()
            base = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(out_dir, f"{base}_{int(time.time())}.gif")

            # 自适应压缩循环
            attempt_fps = int(max(1, fps))
            attempt_width = int(max(64, min(max_width, 4096)))
            min_width = 64
            status = ""

            for _ in range(12):
                ok, msg = self._run_ffmpeg_gif(src, dst, attempt_fps, attempt_width, int(colors), dither, int(loop))
                if not ok:
                    status = msg
                    break
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                if size_mb <= max_size_mb:
                    status = f"OK {size_mb:.2f}MB @ {attempt_fps}fps {attempt_width}px {colors}c dither={dither}"
                    video_name = os.path.basename(dst)
                    video_path_name = os.path.basename(os.path.dirname(dst))
                    _log_info(f"[Video_To_GIF] success {video_name} {size_mb:.2f}MB")
                    return {"ui": {"video": [video_name, video_path_name]}, "result": (dst, status)}
                # 过大则调整：优先降fps，其次降宽度
                if attempt_fps > 6:
                    attempt_fps = max(6, int(attempt_fps * 0.8))
                elif attempt_width > min_width:
                    attempt_width = max(min_width, int(attempt_width * 0.85))
                else:
                    break

            # 若失败，给出状态
            if os.path.exists(dst):
                size_mb = os.path.getsize(dst) / (1024 * 1024)
                status = status or f"Exceeded target: {size_mb:.2f}MB (target {max_size_mb}MB)"
                video_name = os.path.basename(dst)
                video_path_name = os.path.basename(os.path.dirname(dst))
                _log_info(f"[Video_To_GIF] finished with warnings {video_name} {size_mb:.2f}MB")
                return {"ui": {"video": [video_name, video_path_name]}, "result": (dst, status)}
            _log_error(f"[Video_To_GIF] failed: {status}")
            return {"ui": {}, "result": ("", status or "GIF conversion failed")}
        except Exception as e:
            _log_error(f"[Video_To_GIF] exception: {e}")
            return {"ui": {}, "result": ("", f"Error: {str(e)}")}

class PreviewGIFNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gif_path": ("STRING", {"default": ""}),
                "copy_to_output": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Preview a GIF file in browser or external media player"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"

    def _ensure_output_dir(self):
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def preview(self, gif_path, copy_to_output=True):
        if not gif_path or not os.path.exists(gif_path):
            return {"ui": {}, "result": ("",)}

        final_path = gif_path
        if copy_to_output:
            out_dir = self._ensure_output_dir()
            new_path = os.path.join(out_dir, os.path.basename(gif_path))
            if os.path.abspath(new_path) != os.path.abspath(gif_path):
                try:
                    shutil.copy2(gif_path, new_path)
                    final_path = new_path
                except Exception:
                    final_path = gif_path

        # 启用 UI 预览，复用统一前端（video_preview.js）
        video_name = os.path.basename(final_path)
        video_path_name = os.path.basename(os.path.dirname(final_path))
        return {"ui": {"video": [video_name, video_path_name]}, "result": (final_path,)}


    def _open_in_browser(self, file_path):
        """在浏览器中打开文件"""
        try:
            import webbrowser
            import urllib.parse
            # 将文件路径转换为 file:// URL
            file_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(os.path.abspath(file_path)))
            webbrowser.open(file_url)
            _log_info(f"🌐 已在浏览器中打开: {file_path}")
        except Exception as e:
            _log_error(f"❌ 无法在浏览器中打开文件: {e}")

    def _open_in_media_player(self, file_path):
        """在系统默认媒体播放器中打开文件"""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", file_path])
            else:
                # 通用方法
                subprocess.run(["open", file_path])
            
            _log_info(f"🎬 已在媒体播放器中打开: {file_path}")
        except Exception as e:
            _log_error(f"❌ 无法在媒体播放器中打开文件: {e}")

class VideoPreviewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Preview video with clear previous content"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        # 从 VIDEO 对象中提取实际的文件路径
        video_path = extract_video_path(video)

        if not video_path:
            _log_error("❌ VideoPreviewNode: 无法从 VIDEO 对象中提取文件路径")
            return {"ui": {"text": ["Error: Cannot extract video path"]}}

        if not os.path.exists(video_path):
            _log_error(f"❌ VideoPreviewNode: 视频文件不存在: {video_path}")
            return {"ui": {"text": [f"Error: Video file not found: {os.path.basename(video_path)}"]}}

        # 检查视频编码格式，特别是 Topaz 处理的 MPEG-4 视频
        codec_warning = None
        try:
            import subprocess
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                codec_name = probe_result.stdout.strip()
                _log_info(f"🎬 VideoPreviewNode: 检测到视频编码: {codec_name}")

                # 检查是否是 MPEG-4 part 2 编码（Topaz 常用）
                if codec_name == "mpeg4":
                    is_topaz = "topaz" in os.path.basename(video_path).lower()
                    if is_topaz:
                        _log_warning(f"⚠️ VideoPreviewNode: 检测到 Topaz Video AI 处理的 MPEG-4 视频，浏览器预览可能有问题")
                        codec_warning = "topaz_mpeg4"
                    else:
                        _log_warning(f"⚠️ VideoPreviewNode: 检测到 MPEG-4 part 2 编码，浏览器预览可能有问题")
                        codec_warning = "mpeg4"
        except Exception as e:
            _log_warning(f"⚠️ VideoPreviewNode: 无法检测视频编码: {e}")

        video_name = os.path.basename(video_path)
        video_dir = os.path.dirname(video_path)
        video_path_name = os.path.basename(video_dir)

        _log_info(f"🎬 VideoPreviewNode 路径信息:")
        _log_info(f"   - video_path: {video_path}")
        _log_info(f"   - video_dir: {video_dir}")
        _log_info(f"   - video_name: {video_name}")
        _log_info(f"   - video_path_name: {video_path_name}")

        # 构建返回的 UI 数据
        ui_data = {"video": [video_name, video_path_name]}

        # 如果检测到编码问题，添加警告信息
        if codec_warning:
            ui_data["codec_warning"] = codec_warning
            ui_data["video_path"] = video_path  # 添加完整路径用于前端判断

        _log_info(f"🎬 VideoPreviewNode 返回数据: {ui_data}")

        # 使用clear参数来清除之前的视频预览
        result = {"ui": ui_data, "clear": True}
        _log_info(f"🎬 VideoPreviewNode 完整返回: {result}")
        return result

class VideoUtilitiesUploadLiveVideo:
    @classmethod
    def INPUT_TYPES(s):
        # 获取视频文件扩展名
        video_extensions = ["mp4", "webm", "mkv", "avi"]

        # 收集所有视频文件及其修改时间
        all_files = []

        # 扫描 input 目录
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1].lower() in video_extensions:
                    file_path = os.path.join(input_dir, f)
                    mtime = os.path.getmtime(file_path)
                    all_files.append((f, mtime))

        # 扫描 output 目录（跳过同名文件）
        if os.path.exists(output_dir):
            existing_names = {f for f, _ in all_files}
            for f in os.listdir(output_dir):
                if os.path.isfile(os.path.join(output_dir, f)) and f.split('.')[-1].lower() in video_extensions:
                    if f not in existing_names:  # 避免重复
                        file_path = os.path.join(output_dir, f)
                        mtime = os.path.getmtime(file_path)
                        all_files.append((f, mtime))

        # 按修改时间倒序排序（最新的在前）
        all_files.sort(key=lambda x: x[1], reverse=True)

        # 提取文件名（不带前缀）
        files = [f for f, _ in all_files]

        if not files:
            files = ["No video files found"]

        return {"required":{
            "video":(files,),
        },
        "optional":{
            "upload":("VIDEOPLOAD_LIVE",),
        },
        }
    
    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        """验证输入，允许动态文件名（如上传的文件）"""
        # 如果是默认选项，直接通过验证
        if video == "No video files found":
            return True

        # 尝试在 input 和 output 目录中查找文件
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)

        # 如果文件存在于任一目录，验证通过
        if os.path.exists(input_path) or os.path.exists(output_path):
            return True

        # 即使文件不存在也返回 True，允许上传功能
        # 上传后文件会出现在 input 目录
        return True
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        # 只要有参数变化就强制刷新下拉选项
        import random
        return random.random()
    
    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Upload and live video loader with preview functionality"

    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_FILENAMES", "INT", "INT", "STRING",)
    RETURN_NAMES = ("IMAGE", "AUDIO", "FILENAMES", "FILE_AGE", "FRAME_COUNT", "STATUS",)

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video, **kwargs):
        # 兼容直接传入hash文件名
        if video.startswith("[Output] "):
            actual_filename = video[9:]  # 移除 "[Output] " 前缀
            video_path = os.path.join(output_dir, actual_filename)
        elif video.startswith("[Input] "):
            actual_filename = video[8:]  # 移除 "[Input] " 前缀
            video_path = os.path.join(input_dir, actual_filename)
        elif os.path.exists(os.path.join(input_dir, video)):
            actual_filename = video
            video_path = os.path.join(input_dir, video)
        elif os.path.exists(os.path.join(output_dir, video)):
            actual_filename = video
            video_path = os.path.join(output_dir, video)
        elif video.startswith("--- ") or video == "No video files found":
            # 分类标签或无文件情况
            return (None, None, "", 0, "Please select a valid video file")
        else:
            # 兼容旧格式，默认从input目录
            actual_filename = video
            video_path = os.path.join(input_dir, video)
        
        video_filename = os.path.basename(video_path)
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return (None, None, (False, []), 0, f"Video file not found: {video_filename}")
        
        # 计算文件年龄（秒）
        file_age = int(time.time() - os.path.getmtime(video_path))
        
        # 提取音频
        try:
            import subprocess
            # 根据视频来源选择临时文件目录
            temp_dir = output_dir if video.startswith("[Output] ") else input_dir
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as aud:
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    aud.name, "-y"
                ]
                proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0 and os.path.exists(aud.name) and os.path.getsize(aud.name) > 0:
                waveform, sample_rate = torchaudio.load(aud.name)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                audio = None
            # 清理临时音频文件
            try:
                os.unlink(aud.name)
            except:
                pass
        except:
            audio = None
        
        # 提取视频帧作为图像序列
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频的总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[VideoUtilitiesUploadLiveVideo] Video info: {total_frames} frames, {fps:.2f} fps")
            
            frames = []
            frame_count = 0
            
            # 读取所有帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 验证frame_rgb的形状和数据类型
                print(f"[VideoUtilitiesUploadLiveVideo] Frame RGB shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
                
                # 直接转换为torch tensor
                frame_tensor = torch.from_numpy(frame_rgb).float()
                # 转换为0-1范围
                frame_tensor = frame_tensor / 255.0
                
                # 确保维度顺序正确：(H, W, C) -> (C, H, W)
                if frame_tensor.dim() == 3:
                    # 验证输入格式
                    H, W, C = frame_tensor.shape
                    print(f"[VideoUtilitiesUploadLiveVideo] Frame tensor before permute: H={H}, W={W}, C={C}")
                    
                    if C != 3:
                        print(f"[VideoUtilitiesUploadLiveVideo] Error: Expected 3 channels, got {C}")
                        continue
                    
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # (3, H, W)
                    print(f"[VideoUtilitiesUploadLiveVideo] Frame tensor after permute: {frame_tensor.shape}")
                    
                    # 确保数据类型为float32
                    frame_tensor = frame_tensor.float()
                    
                    # 验证值范围
                    if frame_tensor.min() < 0.0 or frame_tensor.max() > 1.0:
                        print(f"[VideoUtilitiesUploadLiveVideo] Warning: Values outside [0,1] range: min={frame_tensor.min():.3f}, max={frame_tensor.max():.3f}")
                        frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                    
                    # 最终验证
                    if frame_tensor.shape[0] != 3:
                        print(f"[VideoUtilitiesUploadLiveVideo] Error: Final tensor should have 3 channels, got {frame_tensor.shape[0]}")
                        continue
                else:
                    print(f"[VideoUtilitiesUploadLiveVideo] Error: Expected 3D tensor (H, W, C), got {frame_tensor.shape}")
                    continue
                
                frames.append(frame_tensor)
                frame_count += 1
            
            cap.release()
            
            if frames:
                # 堆叠所有帧，格式为(batch_size, channels, height, width)
                image_tensor = torch.stack(frames, dim=0)  # (N, 3, H, W) 其中N是实际帧数
                # 确保数据类型为float32
                image_tensor = image_tensor.float()
                
                # 验证tensor格式
                if len(image_tensor.shape) == 4:
                    N, C, H, W = image_tensor.shape
                    print(f"[VideoUtilitiesUploadLiveVideo] Extracted {N} frames, tensor shape: {image_tensor.shape}")
                    print(f"[VideoUtilitiesUploadLiveVideo] Format: (N={N}, C={C}, H={H}, W={W})")
                    
                    # 确保值范围在0-1之间
                    if image_tensor.max() > 1.0 or image_tensor.min() < 0.0:
                        print(f"[VideoUtilitiesUploadLiveVideo] Warning: Values outside [0,1] range: min={image_tensor.min():.3f}, max={image_tensor.max():.3f}")
                        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
                    
                    # 确保tensor格式完全正确
                    if C != 3:
                        print(f"[VideoUtilitiesUploadLiveVideo] Error: Expected 3 channels, got {C}")
                        image_tensor = None
                    elif H <= 0 or W <= 0:
                        print(f"[VideoUtilitiesUploadLiveVideo] Error: Invalid dimensions H={H}, W={W}")
                        image_tensor = None
                    else:
                        # 转换为VideoHelperSuite兼容格式：(N, C, H, W) -> (N, H, W, C)
                        image_tensor = image_tensor.permute(0, 2, 3, 1)  # (N, H, W, C)
                        print(f"[VideoUtilitiesUploadLiveVideo] Converted to VideoHelperSuite format: (N={N}, H={H}, W={W}, C={C})")
                else:
                    print(f"[VideoUtilitiesUploadLiveVideo] Error: Unexpected tensor shape: {image_tensor.shape}")
                    image_tensor = None
            else:
                image_tensor = None
        except Exception as e:
            print(f"[VideoUtilitiesUploadLiveVideo] Error extracting image: {e}")
            image_tensor = None
        
        # 返回结果，包含状态信息
        if video_path:
            # 获取文件大小和创建时间
            try:
                file_size = os.path.getsize(video_path)
                file_size_mb = file_size / (1024 * 1024)
                # 获取文件创建时间
                create_time = time.localtime(os.path.getctime(video_path))
                create_time_str = time.strftime("%Y-%m-%d %H:%M:%S", create_time)
                status = f"Loaded: {video_filename} ({file_size_mb:.1f}MB, created: {create_time_str})"
            except:
                # 如果获取创建时间失败，使用修改时间
                try:
                    modify_time = time.localtime(os.path.getmtime(video_path))
                    modify_time_str = time.strftime("%Y-%m-%d %H:%M:%S", modify_time)
                    status = f"Loaded: {video_filename} (modified: {modify_time_str})"
                except:
                    status = f"Loaded: {video_filename}"
        else:
            status = "No video found"

        # VHS_FILENAMES 兼容输出（与 Video Combine 节点一致）：(bool, [full_paths])
        vhs_filenames = (True, [video_path]) if os.path.exists(video_path) else (False, [])
        frame_count = int(total_frames) if 'total_frames' in locals() else 0
        return (image_tensor, audio, vhs_filenames, file_age, frame_count, status)

class VideoUtilitiesLoadAFVideo:
    @classmethod
    def INPUT_TYPES(s):
        # 扫描 input 和 output 目录及其子文件夹中的视频文件
        files = []
        video_extensions = ["mp4", "webm", "mkv", "avi"]

        # 扫描 input 目录
        if os.path.exists(input_dir):
            # 扫描根目录
            for f in os.listdir(input_dir):
                file_path = os.path.join(input_dir, f)
                if os.path.isfile(file_path) and f.split('.')[-1].lower() in video_extensions:
                    # 不添加前缀，直接使用文件名
                    if f not in files:
                        files.append(f)

            # 扫描子文件夹（一层）
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    for f in os.listdir(item_path):
                        file_path = os.path.join(item_path, f)
                        if os.path.isfile(file_path) and f.split('.')[-1].lower() in video_extensions:
                            # 使用相对路径格式：subfolder/filename
                            relative_path = f"{item}/{f}"
                            if relative_path not in files:
                                files.append(relative_path)

        # 扫描 output 目录
        if os.path.exists(output_dir):
            # 扫描根目录
            for f in os.listdir(output_dir):
                file_path = os.path.join(output_dir, f)
                if os.path.isfile(file_path) and f.split('.')[-1].lower() in video_extensions:
                    # 不添加前缀，但如果 input 目录已有同名文件，则跳过
                    # 这样优先显示 input 目录的文件
                    if f not in files:
                        files.append(f)

            # 扫描子文件夹（一层）
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    for f in os.listdir(item_path):
                        file_path = os.path.join(item_path, f)
                        if os.path.isfile(file_path) and f.split('.')[-1].lower() in video_extensions:
                            # 使用相对路径格式：subfolder/filename
                            relative_path = f"{item}/{f}"
                            if relative_path not in files:
                                files.append(relative_path)

        # 如果没有找到任何视频文件，添加提示信息
        if not files:
            files.append("No video files found")

        return {"required":{
            "video":(files,),
        },
        "optional":{
            "upload":("VIDEOPLOAD",),
        },
        }

    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Load Audio/Video File with preview functionality"

    RETURN_TYPES = ("VIDEO","AUDIO",)

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        """验证输入，允许动态文件名（如上传的文件）"""
        # 如果是默认选项，直接通过验证
        if video == "No video files found":
            return True

        # 尝试在 input 和 output 目录中查找文件
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)

        # 如果文件存在于任一目录，验证通过
        if os.path.exists(input_path) or os.path.exists(output_path):
            return True

        # 即使文件不存在也返回 True，允许上传功能
        # 上传后文件会出现在 input 目录
        return True

    @classmethod
    def IS_CHANGED(s, video):
        # 优先在 input 目录查找，找不到再去 output 目录
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)

        if os.path.exists(input_path):
            return os.path.getmtime(input_path)
        elif os.path.exists(output_path):
            return os.path.getmtime(output_path)

        return float("nan")

    def load_video(self, video):
        # 优先在 input 目录查找，找不到再去 output 目录
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)

        if os.path.exists(input_path):
            video_path = input_path
            base_dir = input_dir
        elif os.path.exists(output_path):
            video_path = output_path
            base_dir = output_dir
        else:
            # 默认使用 input 目录（可能是即将上传的文件）
            video_path = input_path
            base_dir = input_dir

        _log_info(f"🎬 Load_AF_Video: video={video}")
        _log_info(f"🎬 Load_AF_Video: base_dir={base_dir}")
        _log_info(f"🎬 Load_AF_Video: video_path={video_path}")

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            _log_error(f"视频文件不存在: {video_path}")
            return (None, None)

        # 检查视频编码格式，特别是 Topaz 处理的 MPEG-4 视频
        try:
            import subprocess
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                import json
                probe_data = json.loads(probe_result.stdout)
                if probe_data.get("streams"):
                    codec_name = probe_data["streams"][0].get("codec_name", "unknown")
                    _log_info(f"🎬 检测到视频编码: {codec_name}")

                    # 如果是 MPEG-4 part 2 编码（Topaz 常用），记录警告
                    if codec_name == "mpeg4":
                        _log_warning(f"⚠️ 检测到 MPEG-4 part 2 编码（可能来自 Topaz Video AI），某些浏览器可能不支持预览")
        except Exception as e:
            _log_warning(f"⚠️ 无法检测视频编码: {e}")

        # 直接返回路径字符串，以保持与其他节点的兼容性
        # 不使用 VideoFromFile 对象，因为很多节点（如 ProPainter）期望字符串路径
        video_obj = video_path
        _log_info(f"✅ Load AF Video 返回路径字符串: {video_path}")

        try:
            import subprocess
            _log_info(f"🎵 开始提取音频: {video_path}")

            with tempfile.NamedTemporaryFile(suffix=".wav",dir=input_dir,delete=False) as aud:
                temp_audio_path = aud.name

            cmd = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                temp_audio_path, "-y"
            ]
            _log_info(f"🎵 FFmpeg 命令: {' '.join(cmd)}")

            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                _log_error(f"❌ FFmpeg 提取音频失败 (返回码: {proc.returncode})")
                if proc.stderr:
                    _log_error(f"❌ FFmpeg 错误: {proc.stderr}")
                audio = None
            elif not os.path.exists(temp_audio_path):
                _log_error(f"❌ 临时音频文件不存在: {temp_audio_path}")
                audio = None
            elif os.path.getsize(temp_audio_path) == 0:
                _log_warning(f"⚠️ 视频没有音频轨道或音频为空")
                audio = None
            else:
                _log_info(f"✅ 音频提取成功，文件大小: {os.path.getsize(temp_audio_path)} bytes")
                try:
                    waveform, sample_rate = torchaudio.load(temp_audio_path)
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                    _log_info(f"✅ 音频加载成功，采样率: {sample_rate}, 波形形状: {waveform.shape}")
                except Exception as e:
                    _log_error(f"❌ 加载音频文件失败: {e}")
                    audio = None

            # 清理临时文件
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    _log_info(f"🗑️ 已删除临时音频文件")
            except Exception as e:
                _log_warning(f"⚠️ 删除临时文件失败: {e}")

        except Exception as e:
            _log_error(f"❌ 音频提取过程出错: {e}")
            import traceback
            _log_error(traceback.format_exc())
            audio = None
        return (video_obj,audio,)

class VideoUtilitiesPromptTextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "encode"

    CATEGORY = "Ken-Chen/Video_Utilities"

    def encode(self, text):
        return (text,)

class VideoUtilitiesLiveVideoMonitor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "refresh_interval": ("INT", {"default": 2, "min": 1, "max": 60, "step": 1, "tooltip": "Refresh interval in seconds"}),
                "auto_refresh": ("BOOLEAN", {"default": True, "tooltip": "Enable automatic refresh"}),
                "min_file_age": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1, "tooltip": "Minimum file age in seconds to avoid loading incomplete files"}),
            }
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Live video loader that monitors output directory for latest video files"

    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_FILENAMES", "INT", "INT", "STRING",)
    RETURN_NAMES = ("IMAGE", "AUDIO", "FILENAMES", "FILE_AGE", "FRAME_COUNT", "STATUS",)

    OUTPUT_NODE = True

    FUNCTION = "load_latest_video"

    def load_latest_video(self, refresh_interval, auto_refresh, min_file_age):
        video_extensions = ["*.mp4", "*.webm", "*.mkv", "*.avi"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(output_dir, ext)))
            video_files.extend(glob.glob(os.path.join(output_dir, ext.upper())))

        if not video_files:
            return {"result": (None, None, "", 0, "No video files found")}

        latest_video = max(video_files, key=os.path.getmtime)
        video_filename = os.path.basename(latest_video)

        file_age = int(time.time() - os.path.getmtime(latest_video))
        if file_age < min_file_age:
            return {"result": (None, None, "", file_age, f"File too new (age: {file_age}s)")}

        try:
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=output_dir, delete=False) as aud:
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", latest_video, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    aud.name, "-y"
                ]
                proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0 and os.path.exists(aud.name) and os.path.getsize(aud.name) > 0:
                waveform, sample_rate = torchaudio.load(aud.name)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                audio = None
            try:
                os.unlink(aud.name)
            except:
                pass
        except:
            audio = None

        try:
            cap = cv2.VideoCapture(latest_video)
            frames = []
            # 获取总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                if frame_tensor.dim() == 3 and frame_tensor.shape[2] == 3:
                    frame_tensor = frame_tensor.permute(2, 0, 1).float()
                    frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                else:
                    continue
                frames.append(frame_tensor)
            cap.release()

            if frames:
                image_tensor = torch.stack(frames, dim=0).float()
                if len(image_tensor.shape) == 4 and image_tensor.shape[1] == 3:
                    image_tensor = image_tensor.permute(0, 2, 3, 1)
                else:
                    image_tensor = None
            else:
                image_tensor = None
        except Exception:
            image_tensor = None

        if latest_video:
            try:
                file_size_mb = os.path.getsize(latest_video) / (1024 * 1024)
                create_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(latest_video)))
                status = f"Loaded: {video_filename} ({file_size_mb:.1f}MB, created: {create_time_str})"
            except:
                status = f"Loaded: {video_filename}"
        else:
            status = "No video found"

        if latest_video and os.path.exists(latest_video):
            video_path_name = "output"
            vhs_filenames = (1, [latest_video])
            frame_count = int(total_frames) if 'total_frames' in locals() else 0
            return {"ui": {"video": [video_filename, video_path_name]}, "result": (image_tensor, audio, vhs_filenames, file_age, frame_count, status)}
        else:
            vhs_filenames = (0, [])
            frame_count = int(total_frames) if 'total_frames' in locals() else 0
            return {"result": (image_tensor, audio, vhs_filenames, file_age, frame_count, status)}

class VideoUtilitiesRGBEmptyImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "R": ("INT", {"default": 124, "min": 0, "max": 255, "step": 1, "display": "number"}),
                "G": ("INT", {"default": 252, "min": 0, "max": 255, "step": 1, "display": "number"}),
                "B": ("INT", {"default": 0,   "min": 0, "max": 255, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gen_img"
    CATEGORY = "Ken-Chen/Video_Utilities"

    def gen_img(self, width: int, height: int, R: int, G: int, B: int):
        new_image = Image.new("RGB", (width, height), color=(R, G, B))
        tensor = torch.from_numpy(np.asarray(new_image) / 255.0).unsqueeze(0)
        return (tensor,)

class VideoUtilitiesAudioToSubtitle:
    """将视频音频转录成字幕并烧录到视频上"""

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 Fonts 目录下的所有字体文件
        fonts_dir = os.path.join(os.path.dirname(__file__), "Fonts")
        font_files = []

        if os.path.exists(fonts_dir):
            for file in sorted(os.listdir(fonts_dir)):
                if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    font_files.append(file)

        # 如果没有字体文件，添加提示
        if not font_files:
            font_files = ["请将字体文件放入 Fonts 目录"]

        return {
            "required": {
                "video": ("VIDEO",),
                "asr_model": (["SenseVoice", "Whisper"], {"default": "SenseVoice"}),
                "model_size": (["tiny", "base", "small", "medium", "large"], {"default": "medium"}),
                "language": (["auto", "zh", "en", "yue", "ja", "ko", "es", "fr", "de", "ru"], {"default": "auto"}),
                "font_size": ("INT", {"default": 24, "min": 10, "max": 100, "step": 1}),
                "font_file": (font_files, {"default": font_files[0]}),
                "font_color": (["white", "yellow", "black", "red", "green", "blue"], {"default": "yellow"}),
                "text_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "position": (["bottom", "top", "middle"], {"default": "bottom"}),
                "background": (["yes", "no"], {"default": "yes"}),
                "animation": ([
                    "none",           # 无动画
                    "fade_in",        # 淡入
                    "slide_up",       # 从下滑入
                    "slide_down",     # 从上滑入
                    "zoom_in",        # 放大
                    "typewriter",     # 打字机效果
                    "bounce",         # 弹跳
                    "wave",           # 波浪
                    "glow",           # 发光
                    "rainbow",        # 彩虹色
                    "karaoke"         # 卡拉OK效果
                ], {"default": "fade_in"}),
                "animation_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "可选：输入提示文本来引导转录（例如：专业术语、人名、地名等）"
                }),
                "reference_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "可选：输入准确的参考文本，系统会自动校正转录结果（每行一句）"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING",)
    RETURN_NAMES = ("video_with_subtitle", "srt_path",)
    FUNCTION = "add_subtitle"
    CATEGORY = "Ken-Chen/Video_Utilities"
    DESCRIPTION = "Transcribe audio to subtitle and burn it into video using Whisper ASR"
    OUTPUT_NODE = False

    def add_subtitle(self, video, asr_model, model_size, language, font_size, font_file, font_color, text_direction, position, background, animation, animation_duration, prompt="", reference_text=""):
        """将音频转录成字幕并烧录到视频上"""
        try:
            # 提取视频路径
            video_path = extract_video_path(video)
            if not video_path or not os.path.exists(video_path):
                _log_error("❌ 无法获取视频路径或视频文件不存在")
                return (video, "")

            _log_info(f"🎬 开始处理视频: {video_path}")

            # 1. 提取音频
            _log_info("🎵 正在提取音频...")
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                _log_error("❌ 音频提取失败")
                return (video, "")

            # 2. 使用选定的 ASR 模型转录音频
            if asr_model == "SenseVoice":
                result = self._transcribe_sensevoice(audio_path, language, prompt)
            else:  # Whisper
                result = self._transcribe_whisper(audio_path, model_size, language, prompt)

            if not result:
                _log_error("❌ 音频转录失败")
                return (video, "")

            # 2.5. 如果提供了参考文本，进行智能校正
            if reference_text and reference_text.strip():
                _log_info("🔧 正在使用参考文本校正转录结果...")
                result["segments"] = self._correct_segments_with_reference(
                    result["segments"], reference_text
                )

            # 3. 生成 SRT 字幕文件
            _log_info("📝 正在生成 SRT 字幕文件...")
            srt_path = self._generate_srt(video_path, result["segments"])
            if not srt_path:
                _log_error("❌ SRT 字幕生成失败")
                # 清理临时音频文件
                try:
                    os.unlink(audio_path)
                except:
                    pass
                return (video, "")

            _log_info(f"✅ SRT 字幕已保存: {srt_path}")

            # 4. 将字幕烧录到视频上
            _log_info(f"🔥 正在将字幕烧录到视频（方向: {text_direction}, 动画: {animation}）...")
            output_video_path = self._burn_subtitle(
                video_path, srt_path, font_size, font_file, font_color, text_direction, position, background, animation, animation_duration
            )

            # 清理临时音频文件
            try:
                os.unlink(audio_path)
            except:
                pass

            if not output_video_path:
                _log_error("❌ 字幕烧录失败")
                return (video, srt_path)

            _log_info(f"✅ 字幕视频已生成: {output_video_path}")

            # 转换为 ComfyUI VIDEO 对象
            output_video = video_to_comfyui_video(output_video_path)
            if not output_video:
                _log_error("❌ 视频对象转换失败")
                return (video, srt_path)

            return (output_video, srt_path)

        except Exception as e:
            _log_error(f"❌ 字幕处理失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return (video, "")

    def _transcribe_sensevoice(self, audio_path, language, prompt):
        """使用 SenseVoice 转录音频"""
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            _log_error("❌ 未安装 FunASR，请运行: pip install funasr modelscope")
            return None

        try:
            _log_info("🎤 正在使用 SenseVoice 转录音频...")

            # 加载 SenseVoice 模型
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

            # 设置语言参数
            lang_map = {
                "auto": "auto",
                "zh": "zh",
                "en": "en",
                "yue": "yue",  # 粤语
                "ja": "ja",
                "ko": "ko",
            }
            lang = lang_map.get(language, "auto")

            # 转录
            _log_info(f"🔧 SenseVoice 参数: language={lang}, use_itn=True")
            res = model.generate(
                input=audio_path,
                cache={},
                language=lang,
                use_itn=True,  # 使用逆文本归一化（添加标点符号）
                batch_size_s=0,  # 不使用批处理
            )

            if not res or len(res) == 0:
                _log_error("❌ SenseVoice 转录结果为空")
                return None

            # 处理结果
            text = rich_transcription_postprocess(res[0]["text"])
            _log_info(f"✅ SenseVoice 转录完成: {text[:100]}...")

            # 转换为 Whisper 格式的结果（用于兼容后续处理）
            # SenseVoice 返回的是完整文本，我们需要将其分段
            segments = []

            _log_info(f"🔍 SenseVoice 结果键: {res[0].keys()}")

            if "timestamp" in res[0] and res[0]["timestamp"]:
                # 如果有时间戳信息
                _log_info(f"🔍 找到 {len(res[0]['timestamp'])} 个时间戳")
                for i, ts in enumerate(res[0]["timestamp"]):
                    _log_info(f"🔍 时间戳 {i}: {ts}")
                    segments.append({
                        "start": ts[0] / 1000.0,  # 转换为秒
                        "end": ts[1] / 1000.0,
                        "text": ts[2] if len(ts) > 2 else text
                    })
            else:
                # 没有时间戳，智能分割文本
                _log_info("⚠️ SenseVoice 没有返回时间戳，将智能分割文本")
                # 获取音频时长
                import wave
                try:
                    with wave.open(audio_path, 'r') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / float(rate)
                        _log_info(f"🔍 音频时长: {duration:.2f} 秒")
                except:
                    duration = 10.0  # 默认 10 秒

                # 智能分割文本为多个字幕段
                segments = self._split_text_to_segments(text, duration)
                _log_info(f"📝 文本已分割为 {len(segments)} 个字幕段")
                for i, seg in enumerate(segments[:3]):  # 显示前3个
                    _log_info(f"   段{i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text'][:30]}...")

            _log_info(f"📝 生成了 {len(segments)} 个字幕段落")

            return {"segments": segments}

        except Exception as e:
            _log_error(f"❌ SenseVoice 转录失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return None

    def _correct_segments_with_reference(self, segments, reference_text):
        """使用参考文本校正转录结果

        Args:
            segments: ASR 转录的字幕段列表（包含时间戳）
            reference_text: 用户提供的准确文本（每行一句）

        Returns:
            corrected_segments: 校正后的字幕段列表
        """
        import re
        from difflib import SequenceMatcher

        # 解析参考文本（按行分割）
        reference_lines = [line.strip() for line in reference_text.strip().split('\n') if line.strip()]

        _log_info(f"📝 参考文本有 {len(reference_lines)} 行")
        _log_info(f"📝 转录结果有 {len(segments)} 个段落")

        # 提取转录文本（用于对比）
        transcribed_texts = [seg['text'].strip() for seg in segments]

        # 如果参考文本行数与转录段落数相同，直接一一对应
        if len(reference_lines) == len(segments):
            _log_info("✅ 参考文本行数与转录段落数相同，直接替换")
            corrected_segments = []
            for i, seg in enumerate(segments):
                corrected_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": reference_lines[i]
                })
                if transcribed_texts[i] != reference_lines[i]:
                    _log_info(f"   段{i+1} 已校正: '{transcribed_texts[i][:30]}...' → '{reference_lines[i][:30]}...'")
            return corrected_segments

        # 如果数量不同，使用智能匹配
        _log_info("🔍 参考文本行数与转录段落数不同，使用智能匹配...")

        # 合并所有转录文本
        full_transcribed = ''.join(transcribed_texts)
        full_reference = ''.join(reference_lines)

        # 移除空格和标点，用于相似度对比
        def normalize_text(text):
            # 移除所有空格、标点符号
            text = re.sub(r'[，。！？；、,.!?;，\s]', '', text)
            return text.lower()

        norm_transcribed = normalize_text(full_transcribed)
        norm_reference = normalize_text(full_reference)

        # 计算相似度
        similarity = SequenceMatcher(None, norm_transcribed, norm_reference).ratio()
        _log_info(f"📊 文本相似度: {similarity * 100:.1f}%")

        # 如果相似度很高（>80%），使用逐句匹配
        if similarity > 0.8:
            corrected_segments = self._match_segments_to_reference(
                segments, reference_lines, transcribed_texts
            )
        else:
            # 相似度较低，警告用户但仍然尝试匹配
            _log_info(f"⚠️ 文本相似度较低 ({similarity * 100:.1f}%)，可能匹配不准确")
            corrected_segments = self._match_segments_to_reference(
                segments, reference_lines, transcribed_texts
            )

        return corrected_segments

    def _match_segments_to_reference(self, segments, reference_lines, transcribed_texts):
        """智能匹配转录段落和参考文本

        策略：
        1. 如果参考文本行数 <= 转录段落数：合并多个转录段落
        2. 如果参考文本行数 > 转录段落数：拆分转录段落的时间
        """
        import re
        from difflib import SequenceMatcher

        def normalize_text(text):
            return re.sub(r'[，。！？；、,.!?;，\s]', '', text).lower()

        corrected_segments = []

        if len(reference_lines) <= len(segments):
            # 情况1: 参考文本较少，合并转录段落
            _log_info(f"📝 合并模式: {len(segments)} 个段落 → {len(reference_lines)} 行")

            # 计算每行参考文本应该对应多少个转录段落
            segments_per_line = len(segments) / len(reference_lines)

            current_seg_idx = 0
            for i, ref_line in enumerate(reference_lines):
                # 计算这行应该使用多少个转录段落
                start_idx = int(i * segments_per_line)
                end_idx = int((i + 1) * segments_per_line)

                # 确保不超出范围
                end_idx = min(end_idx, len(segments))
                if i == len(reference_lines) - 1:
                    end_idx = len(segments)

                # 合并这些段落的时间范围
                if start_idx < len(segments) and end_idx <= len(segments):
                    start_time = segments[start_idx]["start"]
                    end_time = segments[end_idx - 1]["end"]

                    corrected_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": ref_line
                    })

                    _log_info(f"   段{i+1}: {start_time:.2f}s - {end_time:.2f}s: {ref_line[:30]}...")

        else:
            # 情况2: 参考文本较多，拆分转录段落的时间
            _log_info(f"📝 拆分模式: {len(segments)} 个段落 → {len(reference_lines)} 行")

            # 计算总时长
            total_duration = segments[-1]["end"] - segments[0]["start"]

            # 按参考文本的字符数比例分配时间
            total_chars = sum(len(line) for line in reference_lines)
            current_time = segments[0]["start"]

            for i, ref_line in enumerate(reference_lines):
                # 按字符数比例分配时间
                char_ratio = len(ref_line) / total_chars
                segment_duration = total_duration * char_ratio

                # 确保最后一个段落结束时间等于总时长
                if i == len(reference_lines) - 1:
                    end_time = segments[-1]["end"]
                else:
                    end_time = current_time + segment_duration

                corrected_segments.append({
                    "start": current_time,
                    "end": end_time,
                    "text": ref_line
                })

                _log_info(f"   段{i+1}: {current_time:.2f}s - {end_time:.2f}s: {ref_line[:30]}...")

                current_time = end_time

        _log_info(f"✅ 校正完成，生成 {len(corrected_segments)} 个字幕段")

        return corrected_segments

    def _split_text_to_segments(self, text, duration):
        """智能分割文本为多个字幕段

        Args:
            text: 完整文本
            duration: 音频总时长（秒）

        Returns:
            segments: 字幕段列表
        """
        import re

        _log_info(f"🔍 原始文本: {text[:100]}...")

        # 移除表情符号和特殊标记
        clean_text = re.sub(r'[😊😢😡🎼🎵🎶✨💡🔥⭐🌟💫🎯🎨🎭🎪🎬🎤🎧🎼🎹🎺🎻🥁🎸]', '', text)
        clean_text = clean_text.strip()

        _log_info(f"🔍 清理后文本: {clean_text[:100]}...")

        # 按标点符号分割（中文和英文标点）
        # 注意：中文逗号是 ，（全角），英文逗号是 ,（半角）
        # 保留标点符号在句子末尾
        sentences = re.split(r'([。！？；，、.!?;,])', clean_text)

        _log_info(f"🔍 分割后片段数: {len(sentences)}")
        _log_info(f"🔍 前5个片段: {sentences[:5]}")

        # 重新组合句子和标点
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '。！？；，、.!?;,':
                combined_sentences.append(sentences[i] + sentences[i + 1])
                i += 2
            elif sentences[i].strip():
                combined_sentences.append(sentences[i])
                i += 1
            else:
                i += 1

        # 过滤空句子
        combined_sentences = [s.strip() for s in combined_sentences if s.strip()]

        _log_info(f"🔍 组合后句子数: {len(combined_sentences)}")
        for idx, sent in enumerate(combined_sentences[:3]):
            _log_info(f"   句子{idx+1}: {sent}")

        if not combined_sentences:
            # 如果没有分割成功，返回整段文本
            _log_info("⚠️ 分割失败，返回整段文本")
            return [{
                "start": 0.0,
                "end": duration,
                "text": clean_text
            }]

        # 计算每个句子的时长
        # 策略：平均分配时间，每个句子显示相同的时长
        # 这样更符合实际说话的节奏
        num_sentences = len(combined_sentences)
        segment_duration = duration / num_sentences

        segments = []
        current_time = 0.0

        for i, sentence in enumerate(combined_sentences):
            # 确保最后一个段落结束时间等于总时长
            if i == num_sentences - 1:
                end_time = duration
            else:
                end_time = current_time + segment_duration

            segments.append({
                "start": current_time,
                "end": end_time,
                "text": sentence
            })

            _log_info(f"   段{i+1}: {current_time:.2f}s - {end_time:.2f}s: {sentence[:30]}...")

            current_time = end_time

        _log_info(f"✅ 生成 {len(segments)} 个字幕段（平均每段 {segment_duration:.2f}秒）")

        return segments

    def _transcribe_whisper(self, audio_path, model_size, language, prompt):
        """使用 Whisper 转录音频"""
        try:
            import whisper
        except ImportError:
            _log_error("❌ 未安装 openai-whisper，请运行: pip install openai-whisper")
            return None

        try:
            _log_info(f"🎤 正在使用 Whisper ({model_size}) 转录音频...")
            model = whisper.load_model(model_size)

            # 设置转录参数以提高准确度
            # 注意：word_timestamps 在某些环境下可能与 Triton 库冲突，暂时禁用
            transcribe_options = {
                "verbose": False,
                "temperature": 0.0,
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "initial_prompt": prompt.strip() if prompt and prompt.strip() else None,
                "word_timestamps": False,  # 禁用以避免 Triton 兼容性问题
            }

            # 设置语言参数
            if language != "auto" and language != "yue":  # Whisper 不支持粤语
                transcribe_options["language"] = language
            else:
                transcribe_options["language"] = None

            if prompt and prompt.strip():
                _log_info(f"💡 使用提示文本: {prompt.strip()[:50]}...")
            _log_info(f"🔧 Whisper 参数: beam_size=5, best_of=5, word_timestamps=False")

            result = model.transcribe(audio_path, **transcribe_options)

            # 调试：检查 Whisper 返回结果
            _log_info(f"🔍 Whisper 结果键: {result.keys()}")
            if "segments" in result:
                _log_info(f"✅ Whisper 转录完成: {result['text'][:100]}...")
                _log_info(f"📝 Whisper 生成了 {len(result['segments'])} 个字幕段")
                for i, seg in enumerate(result['segments'][:3]):
                    _log_info(f"   段{i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text'][:30]}...")
            else:
                _log_error("❌ Whisper 结果中没有 segments 字段")

            return result

        except Exception as e:
            _log_error(f"❌ Whisper 转录失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return None

    def _extract_audio(self, video_path):
        """提取视频音频为 WAV 格式"""
        try:
            # 生成临时音频文件路径
            audio_path = os.path.join(
                tempfile.gettempdir(),
                f"audio_{int(time.time())}_{random.randint(1000, 9999)}.wav"
            )

            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # 不处理视频
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz 采样率（Whisper 推荐）
                "-ac", "1",  # 单声道
                "-y",  # 覆盖输出文件
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and os.path.exists(audio_path):
                _log_info(f"✅ 音频提取成功: {audio_path}")

                # 检查音频时长
                try:
                    probe_cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        audio_path
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                    if probe_result.returncode == 0:
                        audio_duration = float(probe_result.stdout.strip())
                        _log_info(f"🎵 音频时长: {audio_duration:.2f} 秒")
                except Exception as e:
                    _log_info(f"⚠️ 无法检查音频时长: {e}")

                return audio_path
            else:
                _log_error(f"❌ FFmpeg 错误: {result.stderr}")
                return None

        except Exception as e:
            _log_error(f"❌ 音频提取异常: {str(e)}")
            return None

    def _generate_srt(self, video_path, segments):
        """生成 SRT 字幕文件"""
        try:
            # 生成 SRT 文件路径（与视频同目录）
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            srt_path = os.path.join(video_dir, f"{video_name}_subtitle.srt")

            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, start=1):
                    # SRT 格式：
                    # 1
                    # 00:00:00,000 --> 00:00:02,000
                    # 字幕文本
                    start_time = self._format_timestamp(segment['start'])
                    end_time = self._format_timestamp(segment['end'])
                    text = segment['text'].strip()

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            return srt_path

        except Exception as e:
            _log_error(f"❌ SRT 生成异常: {str(e)}")
            return None

    def _format_timestamp(self, seconds):
        """将秒数转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _apply_animation(self, animation, progress, subtitle_progress, base_x, base_y, base_color, width, height, text_height):
        """应用字幕动画效果

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
        import math

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
                    int(255 * (1 - t) + 0 * t),      # R
                    int(255 * (1 - t) + 255 * t),    # G
                    int(255 * (1 - t) + 255 * t)     # B
                )
            else:
                # 后半段：黄色到红色
                t = (subtitle_progress - 0.5) * 2
                color = (
                    int(0 * (1 - t) + 0 * t),        # R
                    int(255 * (1 - t) + 0 * t),      # G
                    int(255 * (1 - t) + 255 * t)     # B
                )

        return x, y, alpha, scale, color

    def _draw_vertical_text(self, draw, text, font, x, y, color, alpha):
        """绘制竖排文字 - 使用固定间距"""
        # 使用字体大小作为行高，确保间距一致
        try:
            line_height = font.size
        except:
            # 如果无法获取字体大小，使用 bbox 估算
            bbox = draw.textbbox((0, 0), "测", font=font)
            line_height = bbox[3] - bbox[1]

        current_y = y
        for char in text:
            draw.text((x, current_y), char, font=font, fill=(*color, int(255 * alpha)))
            current_y += line_height

    def _get_vertical_text_size(self, draw, text, font):
        """计算竖排文字的尺寸 - 使用固定间距"""
        # 使用字体大小作为行高
        try:
            line_height = font.size
        except:
            # 如果无法获取字体大小，使用 bbox 估算
            bbox = draw.textbbox((0, 0), "测", font=font)
            line_height = bbox[3] - bbox[1]

        total_height = line_height * len(text)

        # 计算最大宽度
        max_width = 0
        for char in text:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            max_width = max(max_width, char_width)

        return max_width, total_height

    def _burn_subtitle(self, video_path, srt_path, font_size, font_file, font_color, text_direction, position, background, animation="none", animation_duration=0.3):
        """将字幕烧录到视频上 - 使用 Python + OpenCV 方法，支持动画效果和竖排文字"""
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            import re

            # 生成输出视频路径
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(video_dir, f"{video_name}_with_subtitle.mp4")

            _log_info(f"🎬 使用 OpenCV 方法烧录字幕...")

            # 读取 SRT 文件
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            # 解析 SRT
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n)*?)(?=\n\d+\n|\Z)'
            matches = re.findall(pattern, srt_content, re.MULTILINE)

            # 转换时间戳为毫秒
            def time_to_ms(time_str):
                h, m, s = time_str.split(':')
                s, ms = s.split(',')
                return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

            # 构建字幕时间轴
            subtitles = []
            for match in matches:
                start_ms = time_to_ms(match[1])
                end_ms = time_to_ms(match[2])
                text = match[3].strip().replace('\n', ' ')
                subtitles.append((start_ms, end_ms, text))

            _log_info(f"📝 解析到 {len(subtitles)} 条字幕")
            if len(subtitles) > 0:
                _log_info(f"📝 第一条字幕: {subtitles[0][0]}ms - {subtitles[0][1]}ms: {subtitles[0][2][:50]}...")
                if len(subtitles) > 1:
                    _log_info(f"📝 第二条字幕: {subtitles[1][0]}ms - {subtitles[1][1]}ms: {subtitles[1][2][:50]}...")

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _log_error(f"❌ 无法打开视频文件: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            _log_info(f"📹 视频信息: {width}x{height} @ {fps}fps, {total_frames} 帧")

            if width == 0 or height == 0 or fps == 0:
                _log_error(f"❌ 视频参数无效: {width}x{height} @ {fps}fps")
                cap.release()
                return None

            # 创建临时输出路径（使用 AVI 格式，更稳定）
            temp_output = os.path.join(video_dir, f"{video_name}_temp.avi")

            # 创建视频写入器 - 使用 XVID 编码器（更可靠）
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                _log_error("❌ 无法创建视频写入器")
                return None

            # 设置字体路径（从 Fonts 目录）
            fonts_dir = os.path.join(os.path.dirname(__file__), "Fonts")
            font_path = os.path.join(fonts_dir, font_file)

            if os.path.exists(font_path):
                _log_info(f"🔤 字体: {font_file}")
            else:
                _log_info(f"⚠️ 字体文件不存在: {font_path}，将使用 PIL 默认字体")
                font_path = None

            # 设置字体颜色（RGB 格式，用于 PIL）
            # 注意：PIL 使用 RGB 格式，不是 BGR！
            color_map = {
                "white": (255, 255, 255),
                "yellow": (255, 255, 0),    # RGB: 红+绿=黄
                "black": (0, 0, 0),
                "red": (255, 0, 0),         # RGB: 红
                "green": (0, 255, 0),       # RGB: 绿
                "blue": (0, 0, 255)         # RGB: 蓝
            }
            text_color = color_map.get(font_color, (255, 255, 0))  # 默认黄色
            _log_info(f"🎨 字幕颜色: {font_color} = RGB{text_color}")

            # 设置字幕位置
            if position == "bottom":
                y_offset = int(height * 0.85)
            elif position == "top":
                y_offset = int(height * 0.15)
            else:  # middle
                y_offset = int(height * 0.5)

            frame_count = 0
            subtitle_found_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 计算当前时间（毫秒）
                current_ms = int((frame_count / fps) * 1000)

                # 查找当前时间的字幕
                current_text = None
                subtitle_start_ms = None
                subtitle_end_ms = None
                for start_ms, end_ms, text in subtitles:
                    if start_ms <= current_ms <= end_ms:
                        current_text = text
                        subtitle_start_ms = start_ms
                        subtitle_end_ms = end_ms
                        if subtitle_found_count < 5:  # 只记录前5次
                            _log_info(f"🎯 帧 {frame_count} ({current_ms}ms) 匹配字幕: {start_ms}-{end_ms}ms: {text[:30]}...")
                            subtitle_found_count += 1
                        break

                # 如果有字幕，绘制到帧上
                if current_text:
                    # 计算动画进度（0.0 到 1.0）
                    subtitle_duration_ms = max(subtitle_end_ms - subtitle_start_ms, 1)  # 避免除零错误
                    elapsed_ms = current_ms - subtitle_start_ms
                    animation_progress = min(elapsed_ms / (animation_duration * 1000), 1.0)  # 动画进度
                    subtitle_progress = min(elapsed_ms / subtitle_duration_ms, 1.0)  # 字幕整体进度

                    # 使用 PIL 绘制中文字幕
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img, 'RGBA')

                    # 加载字体
                    if font_path and os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                        except Exception as e:
                            _log_info(f"⚠️ 无法加载字体 {font_path}: {e}，使用默认字体")
                            font = ImageFont.load_default()
                    else:
                        font = ImageFont.load_default()

                    # 计算文字位置（根据方向）
                    if text_direction == "vertical":
                        # 竖排：计算竖排文字尺寸
                        text_width, text_height = self._get_vertical_text_size(draw, current_text, font)
                        # 竖排时，x 位置靠右，y 位置居中
                        if position == "bottom":
                            base_x = int(width * 0.9) - text_width // 2
                        elif position == "top":
                            base_x = int(width * 0.9) - text_width // 2
                        else:  # middle
                            base_x = int(width * 0.9) - text_width // 2
                        base_y = (height - text_height) // 2
                    else:
                        # 横排：计算横排文字尺寸
                        bbox = draw.textbbox((0, 0), current_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        base_x = (width - text_width) // 2
                        base_y = y_offset - text_height // 2

                    # 应用动画效果
                    x, y, alpha, scale, color = self._apply_animation(
                        animation, animation_progress, subtitle_progress,
                        base_x, base_y, text_color, width, height, text_height
                    )

                    # 绘制背景（如果需要）
                    if background == "yes":
                        padding = 10
                        bg_alpha = int(128 * alpha)
                        draw.rectangle(
                            [x - padding, y - padding, x + text_width * scale + padding, y + text_height * scale + padding],
                            fill=(0, 0, 0, bg_alpha)
                        )

                    # 绘制文字（根据方向）
                    if text_direction == "vertical":
                        # 竖排文字
                        if scale != 1.0 or alpha != 1.0:
                            # 创建临时图像用于缩放
                            temp_img = Image.new('RGBA', (int(text_width * 1.5), int(text_height * 1.5)), (0, 0, 0, 0))
                            temp_draw = ImageDraw.Draw(temp_img)
                            # 竖排绘制
                            self._draw_vertical_text(temp_draw, current_text, font, int(text_width * 0.25), int(text_height * 0.25), color, alpha)

                            if scale != 1.0:
                                new_size = (int(temp_img.width * scale), int(temp_img.height * scale))
                                temp_img = temp_img.resize(new_size, Image.Resampling.LANCZOS)

                            # 粘贴到主图像
                            pil_img.paste(temp_img, (int(x - text_width * 0.25 * scale), int(y - text_height * 0.25 * scale)), temp_img)
                        else:
                            # 直接绘制竖排文字
                            self._draw_vertical_text(draw, current_text, font, x, y, color, alpha)
                    else:
                        # 横排文字
                        if scale != 1.0 or alpha != 1.0:
                            # 创建临时图像用于缩放
                            temp_img = Image.new('RGBA', (int(text_width * 1.5), int(text_height * 1.5)), (0, 0, 0, 0))
                            temp_draw = ImageDraw.Draw(temp_img)
                            temp_draw.text((int(text_width * 0.25), int(text_height * 0.25)), current_text, font=font, fill=(*color, int(255 * alpha)))

                            if scale != 1.0:
                                new_size = (int(temp_img.width * scale), int(temp_img.height * scale))
                                temp_img = temp_img.resize(new_size, Image.Resampling.LANCZOS)

                            # 粘贴到主图像
                            pil_img.paste(temp_img, (int(x - text_width * 0.25 * scale), int(y - text_height * 0.25 * scale)), temp_img)
                        else:
                            # 直接绘制横排文字
                            draw.text((x, y), current_text, font=font, fill=(*color, int(255 * alpha)))

                    # 转换回 OpenCV 格式
                    frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

                out.write(frame)
                frame_count += 1

                # 显示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    _log_info(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

            cap.release()
            out.release()

            _log_info(f"✅ 临时视频已生成: {temp_output}")
            _log_info(f"📹 实际写入帧数: {frame_count}")
            _log_info(f"📹 预期帧数: {total_frames}")
            _log_info(f"📹 视频时长: {frame_count / fps:.2f} 秒")

            # 检查帧数是否匹配
            if abs(frame_count - total_frames) > 5:
                _log_info(f"⚠️ 警告: 写入帧数 ({frame_count}) 与预期 ({total_frames}) 不匹配")

            # 使用 FFmpeg 转换为 H.264 MP4（保留音频）
            _log_info(f"🔄 正在转换为 H.264 格式...")

            # 修复：使用 -shortest 确保视频和音频时长一致
            # 使用 -async 1 确保音视频同步
            # 优化流媒体播放：faststart, 关键帧间隔
            cmd = [
                "ffmpeg",
                "-i", temp_output,     # 带字幕的视频
                "-i", video_path,      # 原始视频（用于提取音频）
                "-map", "0:v:0",       # 使用第一个输入的视频流
                "-map", "1:a:0?",      # 使用第二个输入的音频流（如果存在）
                "-c:v", "libx264",     # H.264 编码
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",         # 重新编码音频（而不是 copy）
                "-b:a", "192k",        # 音频比特率
                "-shortest",           # 使用最短的流作为输出时长
                "-async", "1",         # 音视频同步
                "-vsync", "cfr",       # 恒定帧率
                "-movflags", "+faststart",  # 优化流媒体播放
                "-g", "30",            # 关键帧间隔（每秒一个关键帧）
                "-keyint_min", "30",   # 最小关键帧间隔
                "-sc_threshold", "0",  # 禁用场景切换检测
                "-y",
                output_path
            ]

            _log_info(f"🎬 FFmpeg 命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # 清理临时文件
            try:
                os.remove(temp_output)
            except:
                pass

            if result.returncode == 0 and os.path.exists(output_path):
                _log_info(f"✅ 字幕烧录成功: {output_path}")
                # 检查输出文件大小
                file_size = os.path.getsize(output_path)
                _log_info(f"📊 输出文件大小: {file_size / 1024 / 1024:.2f} MB")
                if file_size < 1024:
                    _log_error(f"⚠️ 输出文件太小 ({file_size} bytes)，可能有问题")

                # 检查输出视频时长
                try:
                    check_cap = cv2.VideoCapture(output_path)
                    if check_cap.isOpened():
                        output_fps = check_cap.get(cv2.CAP_PROP_FPS)
                        output_frames = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        output_duration = output_frames / output_fps if output_fps > 0 else 0
                        _log_info(f"📹 输出视频: {output_frames} 帧 @ {output_fps:.2f}fps = {output_duration:.2f} 秒")

                        # 与原始视频对比
                        original_duration = total_frames / fps if fps > 0 else 0
                        _log_info(f"📹 原始视频: {total_frames} 帧 @ {fps:.2f}fps = {original_duration:.2f} 秒")

                        if abs(output_duration - original_duration) > 0.5:
                            _log_info(f"⚠️ 警告: 输出视频时长 ({output_duration:.2f}s) 与原始 ({original_duration:.2f}s) 不匹配")
                        check_cap.release()
                except Exception as e:
                    _log_info(f"⚠️ 无法检查输出视频时长: {e}")

                return output_path
            else:
                _log_error(f"❌ FFmpeg 转换失败 (返回码: {result.returncode})")
                _log_error(f"❌ FFmpeg stderr: {result.stderr}")
                _log_error(f"❌ FFmpeg stdout: {result.stdout}")
                # 如果转换失败，返回临时文件
                if os.path.exists(temp_output):
                    os.rename(temp_output, output_path)
                    _log_info(f"⚠️ 使用临时文件作为输出: {output_path}")
                    return output_path
                return None

        except Exception as e:
            _log_error(f"❌ 字幕烧录异常: {str(e)}")
            import traceback
            _log_error(f"❌ 详细错误: {traceback.format_exc()}")
            return None

    def _convert_srt_to_ass(self, srt_path, ass_path, font_size, font_color, position, background):
        """将 SRT 字幕转换为 ASS 格式并应用样式"""
        try:
            # 读取 SRT 文件
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            # 解析 SRT
            import re
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n)*?)(?=\n\d+\n|\Z)'
            matches = re.findall(pattern, srt_content, re.MULTILINE)

            # 设置对齐方式
            if position == "bottom":
                alignment = 2  # 底部居中
                margin_v = 20
            elif position == "top":
                alignment = 8  # 顶部居中
                margin_v = 20
            else:  # middle
                alignment = 5  # 中间居中
                margin_v = 0

            # 获取颜色代码
            color_code = self._get_color_code(font_color)

            # 设置背景
            if background == "yes":
                outline = 1
                shadow = 2
                back_color = "&H80000000"  # 半透明黑色
            else:
                outline = 1
                shadow = 0
                back_color = "&H00000000"  # 透明

            # 创建 ASS 文件头
            ass_content = f"""[Script Info]
Title: Generated Subtitle
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H{color_code},&H000000FF,&H00000000,{back_color},0,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

            # 转换时间格式从 SRT 到 ASS
            def srt_time_to_ass(srt_time):
                # SRT: 00:00:00,000 -> ASS: 0:00:00.00
                return srt_time.replace(',', '.')[:-1]

            # 添加字幕事件
            for match in matches:
                start_time = srt_time_to_ass(match[1])
                end_time = srt_time_to_ass(match[2])
                text = match[3].strip().replace('\n', '\\N')
                ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"

            # 写入 ASS 文件
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)

            _log_info(f"✅ ASS 字幕已生成: {ass_path}")

        except Exception as e:
            _log_error(f"❌ SRT 转 ASS 失败: {str(e)}")
            raise

    def _get_color_code(self, color_name):
        """获取 ASS 字幕颜色代码（AABBGGRR 格式）"""
        color_map = {
            "white": "FFFFFF",
            "yellow": "00FFFF",
            "black": "000000",
            "red": "0000FF",
            "green": "00FF00",
            "blue": "FF0000",
        }
        return color_map.get(color_name, "00FFFF")  # 默认黄色

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoStitchingNode": VideoStitchingNode,
    "GetLastFrameNode": GetLastFrameNode,
    "GetFirstFrameNode": GetFirstFrameNode,
    "VideoUtilitiesGetVHSFilePath": VideoUtilitiesGetVHSFilePath,
    "VideoToGIFNode": VideoToGIFNode,
    "PreviewGIFNode": PreviewGIFNode,
    "VideoPreviewNode": VideoPreviewNode,
    "VideoUtilitiesUploadLiveVideo": VideoUtilitiesUploadLiveVideo,
    "VideoUtilitiesLoadAFVideo": VideoUtilitiesLoadAFVideo,
    "VideoUtilitiesPromptTextNode": VideoUtilitiesPromptTextNode,
    "VideoUtilitiesLiveVideoMonitor": VideoUtilitiesLiveVideoMonitor,
    "VideoUtilitiesRGBEmptyImage": VideoUtilitiesRGBEmptyImage,
    "VideoUtilitiesAudioToSubtitle": VideoUtilitiesAudioToSubtitle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStitchingNode": "Video_Stitching",
    "GetLastFrameNode": "Get_Last_Frame",
    "GetFirstFrameNode": "Get_First_Frame",
    "VideoUtilitiesGetVHSFilePath": "Get VHS File Path",
    "VideoToGIFNode": "Video_To_GIF",
    "PreviewGIFNode": "Preview_GIF",
    "VideoPreviewNode": "Video_Preview",
    "VideoUtilitiesUploadLiveVideo": "Upload_Live_Video",
    "VideoUtilitiesLoadAFVideo": "Load_AF_Video",
    "VideoUtilitiesPromptTextNode": "Prompt_Text_Node",
    "VideoUtilitiesLiveVideoMonitor": "Live_Video_Monitor",
    "VideoUtilitiesRGBEmptyImage": "RGB_Empty_Image",
    "VideoUtilitiesAudioToSubtitle": "Audio_To_Subtitle",
}

