"""
Audio To Text Node
语音识别节点
"""

import os
import sys
import tempfile
import subprocess
import time
import random

from ..utils.asr_engine import ASREngine


def _log_info(message):
    print(f"[Audio_To_Text] {message}")


def _log_error(message):
    print(f"[Audio_To_Text ERROR] {message}")


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


class VideoUtilitiesAudioToText:
    """语音识别节点"""

    # 模型列表
    MODELS_LIST = [
        # CTranslate2 格式（推荐，无 torchcodec 依赖）
        "Belle-whisper-large-v3-zh-punct-ct2",  # 推荐：中文优化，带标点
        "Belle-whisper-large-v3-zh-punct-ct2-float32",  # 高精度中文
        "whisper-large-v3-ct2",  # 多语言支持

        # Transformers 格式（需要 torchcodec，可能在 Windows 上有问题）
        "Belle-whisper-large-v3-zh-punct",  # 中文优化，带标点（Transformers）
        "Belle-whisper-large-v3-zh",  # 中文优化（Transformers）
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS_LIST, {"default": cls.MODELS_LIST[0], "tooltip": "选择ASR模型，中文推荐使用 zh 模型"}),
                "max_sentence_length": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1, "tooltip": "中文按字数计算，英文按字母数计算"}),
                "unload_model": ("BOOLEAN", {"default": True, "tooltip": "运行后卸载模型以释放显存"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "随机种子"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "可选：输入视频文件"}),
                "audio": ("AUDIO", {"tooltip": "可选：输入音频文件"}),
                "prompt": ("STRING", {"multiline": False, "default": "", "placeholder": "可选：输入提示文本来引导转录（例如：专业术语、人名、地名等）"}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "placeholder": "可选：输入准确的参考文本，系统会自动校正转录结果（每行一句）"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("纯文本", "逐词时间戳", "逐句时间戳")
    FUNCTION = "recognize"
    CATEGORY = "Video_Utilities/ASR"

    def __init__(self):
        self.asr_engine = ASREngine()
    
    def recognize(
        self,
        model,
        max_sentence_length,
        unload_model,
        seed,
        video=None,
        audio=None,
        prompt="",
        reference_text=""
    ):
        """
        语音识别（支持视频或音频输入）

        Returns:
            (纯文本, 逐词时间戳, 逐句时间戳)
        """
        try:
            # 设置随机种子
            if seed != 0:
                import torch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # 1. 检查输入
            if video is None and audio is None:
                _log_error("❌ 必须提供视频或音频输入")
                return ("", "", "")

            if video is not None and audio is not None:
                _log_error("❌ 不能同时提供视频和音频输入，请只选择一个")
                return ("", "", "")

            # 2. 获取音频文件路径
            audio_path = None
            need_cleanup_audio = False  # 标记是否需要清理临时音频文件

            if video is not None:
                # 从视频中提取音频
                video_path = extract_video_path(video)
                if not video_path or not os.path.exists(video_path):
                    _log_error("❌ 无法获取视频路径或视频文件不存在")
                    return ("", "", "")

                _log_info(f"🎬 开始处理视频: {video_path}")
                _log_info("🎵 正在提取音频...")
                audio_path = self._extract_audio(video_path)
                if not audio_path:
                    _log_error("❌ 音频提取失败")
                    return ("", "", "")

                need_cleanup_audio = True  # 从视频提取的音频需要清理

            elif audio is not None:
                # 处理音频输入
                # AUDIO 类型是一个字典: {"waveform": tensor, "sample_rate": int}
                # 需要将其保存为临时 WAV 文件
                if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                    _log_info("🎵 检测到音频张量，正在保存为临时文件...")
                    import torchaudio

                    # 生成临时音频文件路径
                    audio_path = os.path.join(
                        tempfile.gettempdir(),
                        f"audio_{int(time.time())}_{random.randint(1000, 9999)}.wav"
                    )

                    # 保存音频
                    waveform = audio["waveform"].squeeze(0)  # 移除 batch 维度
                    sample_rate = audio["sample_rate"]
                    torchaudio.save(audio_path, waveform, sample_rate)

                    _log_info(f"🎵 音频已保存到: {audio_path}")

                    # 标记需要清理临时文件
                    need_cleanup_audio = True
                else:
                    # 尝试作为文件路径处理
                    audio_path = extract_video_path(audio)
                    if not audio_path or not os.path.exists(audio_path):
                        _log_error(f"❌ 无法获取音频路径或音频文件不存在，audio 类型: {type(audio)}")
                        return ("", "", "")

                    _log_info(f"🎵 开始处理音频: {audio_path}")
                    need_cleanup_audio = False

            # 3. 语音识别（使用新的模型参数）
            _log_info(f"🎤 正在使用模型 {model} 识别音频...")
            words_list, sentences_list = self.asr_engine.recognize(
                audio_path,
                "faster-whisper",  # 固定使用 faster-whisper 引擎
                model,  # 传递模型名称
                "auto",  # 自动检测语言
                prompt,
                max_sentence_length
            )

            # 清理临时音频文件（如果需要）
            if need_cleanup_audio:
                try:
                    os.unlink(audio_path)
                    _log_info(f"🗑️ 已清理临时音频文件: {audio_path}")
                except Exception as e:
                    _log_error(f"⚠️ 清理临时音频文件失败: {e}")
            
            # 4. 应用参考文本校正（如果提供）
            if reference_text and reference_text.strip():
                _log_info("🔧 正在使用参考文本校正识别结果...")
                sentences_list = self._correct_with_reference(sentences_list, reference_text)
                # 重新生成逐词时间戳（使用 jieba 分词）
                _log_info("🔧 重新生成逐词时间戳...")
                words_list = self.asr_engine._generate_words_from_sentences(sentences_list)

            # 5. 格式化输出
            plain_text = self._format_plain_text(sentences_list)
            words_timestamps = self._format_words_timestamps(words_list)
            sentences_timestamps = self._format_sentences_timestamps(sentences_list)
            
            _log_info(f"✅ 识别完成: {len(sentences_list)} 个句子, {len(words_list)} 个词")

            # 6. 卸载模型（如果需要）
            if unload_model:
                self.asr_engine.unload_model()

            return (plain_text, words_timestamps, sentences_timestamps)
        
        except Exception as e:
            _log_error(f"❌ 识别失败: {str(e)}")
            import traceback
            _log_error(traceback.format_exc())
            return ("", "", "")
    
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
                return audio_path
            else:
                _log_error(f"❌ FFmpeg 错误: {result.stderr}")
                return None
        
        except Exception as e:
            _log_error(f"❌ 音频提取失败: {str(e)}")
            return None
    
    def _correct_with_reference(self, sentences_list, reference_text):
        """
        使用参考文本校正识别结果

        **完全以参考文本为准**：
        - 参考文本有多少行，就生成多少个句子
        - 时间戳根据识别结果的总时长按字符数比例分配

        Args:
            sentences_list: 原始识别结果 [(start, end, text), ...]
            reference_text: 参考文本（多行，每行一个句子）
                          支持延长显示时长：在行末添加 -数字s，例如 "这是一句话 -0.5s" 表示延长0.5秒

        Returns:
            corrected_sentences_list: 校正后的句子列表
        """
        import re

        # 1. 解析参考文本（按行分割，并提取延长时长）
        reference_lines = []
        extend_durations = {}  # {行索引: 延长时长}

        for idx, line in enumerate(reference_text.strip().split('\n')):
            line = line.strip()
            if not line:
                continue

            # 检查是否有延长时长标记（格式：-数字s）
            duration_match = re.search(r'[\s，。！？；：、]*-(\d+(?:\.\d+)?)s\s*$', line)
            if duration_match:
                extend_duration = float(duration_match.group(1))
                # 移除时长标记，保留文本
                line_text = line[:duration_match.start()].strip()
                extend_durations[len(reference_lines)] = extend_duration
                _log_info(f"📝 检测到延长时长: 第 {len(reference_lines) + 1} 句延长 {extend_duration} 秒")
            else:
                line_text = line

            if line_text:  # 确保不为空
                reference_lines.append(line_text)

        if not reference_lines:
            _log_info("⚠️ 参考文本为空，跳过校正")
            return sentences_list

        # 2. 获取识别结果的总时长
        total_start = sentences_list[0][0]
        total_end = sentences_list[-1][1]
        total_duration = total_end - total_start

        _log_info(f"📊 参考文本: {len(reference_lines)} 行")
        _log_info(f"📊 识别结果: {len(sentences_list)} 个句子")
        _log_info(f"📊 总时长: {total_duration:.2f}s ({total_start:.2f}s - {total_end:.2f}s)")

        # 3. 按字符数比例分配时间戳
        total_chars = sum(len(line) for line in reference_lines)
        aligned_sentences = []
        current_time = total_start

        for idx, line in enumerate(reference_lines):
            line_start = current_time

            # 先按字符数计算基础时长
            base_duration = (len(line) / total_chars) * total_duration

            # 检查是否有延长时长
            if idx in extend_durations:
                extend_duration = extend_durations[idx]
                line_duration = base_duration + extend_duration
                _log_info(f"⏱️ 第 {idx+1} 句延长 {extend_duration} 秒 (基础: {base_duration:.2f}s → 延长后: {line_duration:.2f}s)")
            else:
                line_duration = base_duration

            line_end = current_time + line_duration
            aligned_sentences.append((round(line_start, 2), round(line_end, 2), line))
            current_time = line_end

        _log_info(f"✅ 参考文本校正完成: {len(aligned_sentences)} 个句子")
        return aligned_sentences


    def _format_plain_text(self, sentences_list):
        """格式化纯文本"""
        if not sentences_list:
            return ""
        return "\n".join([text for _, _, text in sentences_list])
    
    def _format_words_timestamps(self, words_list):
        """格式化逐词时间戳"""
        if not words_list:
            return ""
        lines = []
        for start, end, word in words_list:
            lines.append(f"({start}, {end}) {word}")
        return "\n".join(lines)
    
    def _format_sentences_timestamps(self, sentences_list):
        """格式化逐句时间戳"""
        if not sentences_list:
            return ""
        lines = []
        for start, end, text in sentences_list:
            lines.append(f"({start}, {end}) {text}")
        return "\n".join(lines)

