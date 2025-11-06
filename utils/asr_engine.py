"""
ASR Engine Wrapper
支持 faster-whisper 引擎，使用 ComfyUI/models/TTS 目录下的模型
"""

import os
import re
import torch
import tempfile
from typing import List, Tuple, Optional

# 日志函数
def _log_info(message):
    print(f"[ASR] {message}")

def _log_error(message):
    print(f"[ASR ERROR] {message}")

def _log_warning(message):
    print(f"[ASR WARNING] ⚠️ {message}")


def _download_model(model_name: str, model_path: str) -> bool:
    """
    自动下载模型

    Args:
        model_name: 模型名称
        model_path: 模型保存路径

    Returns:
        是否下载成功
    """
    # 模型仓库映射
    MODEL_REPOS = {
        # CTranslate2 格式（推荐）
        "Belle-whisper-large-v3-zh-punct-ct2": "k1nto/Belle-whisper-large-v3-zh-punct-ct2",
        "Belle-whisper-large-v3-zh-punct-ct2-float32": "k1nto/Belle-whisper-large-v3-zh-punct-ct2-float32",
        "whisper-large-v3-ct2": "Systran/faster-whisper-large-v3",
        "whisper-medium-ct2": "Systran/faster-whisper-medium",
        "whisper-small-ct2": "Systran/faster-whisper-small",

        # Transformers 格式（需要 torchcodec）
        "Belle-whisper-large-v3-zh-punct": "BELLE-2/Belle-whisper-large-v3-zh-punct",
        "Belle-whisper-large-v3-zh": "BELLE-2/Belle-whisper-large-v3-zh",
    }

    if model_name not in MODEL_REPOS:
        _log_warning(f"未知模型: {model_name}，无法自动下载")
        _log_warning(f"支持的模型: {', '.join(MODEL_REPOS.keys())}")
        return False

    repo_id = MODEL_REPOS[model_name]

    try:
        _log_info(f"📥 开始自动下载模型: {model_name}")
        _log_info(f"📦 仓库: {repo_id}")
        _log_info(f"📁 保存路径: {model_path}")
        _log_info(f"⏳ 这可能需要几分钟到几十分钟，取决于网速...")

        # 尝试导入 huggingface_hub
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            _log_warning("huggingface_hub 未安装，正在安装...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
            from huggingface_hub import snapshot_download

        # 检查是否使用镜像站点
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from config import USE_HF_MIRROR, HF_MIRROR_ENDPOINT, DOWNLOAD_TIMEOUT

            if USE_HF_MIRROR:
                _log_info(f"🌐 使用镜像站点: {HF_MIRROR_ENDPOINT}")
                os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
        except ImportError:
            # 如果配置文件不存在，使用默认值
            DOWNLOAD_TIMEOUT = 3600

        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        _log_info(f"✅ 模型下载成功: {model_name}")
        return True

    except Exception as e:
        _log_warning(f"模型下载失败: {str(e)}")
        _log_warning(f"请手动下载模型:")
        _log_warning(f"  pip install huggingface-hub")
        _log_warning(f"  huggingface-cli download {repo_id} --local-dir {model_path}")
        return False


class ASREngine:
    """ASR 引擎封装类"""

    def __init__(self):
        self.model_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_name = None

        # 获取 ComfyUI models 目录
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            self.model_path = os.path.join(models_dir, "TTS")
            _log_info(f"📁 模型目录: {self.model_path}")
        except Exception as e:
            _log_error(f"无法获取 models 目录: {e}")
            self.model_path = None
    
    def recognize(
        self,
        audio_path: str,
        engine: str,
        model_name: str,
        language: str,
        prompt: Optional[str] = None,
        max_sentence_length: int = 20
    ) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """
        语音识别

        Args:
            audio_path: 音频文件路径
            engine: ASR 引擎（固定为 faster-whisper）
            model_name: 模型名称（如 Belle-whisper-large-v3-zh-punct-ct2）
            language: 语言代码
            prompt: 提示文本
            max_sentence_length: 最大句子长度

        Returns:
            (words_list, sentences_list)
            words_list: [(start, end, word), ...]
            sentences_list: [(start, end, sentence), ...]
        """
        _log_info(f"🎤 ASR 引擎: {engine}, 模型: {model_name}, 语言: {language}")

        # 判断模型格式：CTranslate2 或 Transformers
        # CTranslate2 格式的模型名称包含 "-ct2"（可能后面还有其他后缀，如 -float32）
        if "-ct2" in model_name:
            # CTranslate2 格式（faster-whisper）
            return self._recognize_faster_whisper(audio_path, model_name, language, prompt, max_sentence_length)
        else:
            # Transformers 格式（原生 Whisper）
            return self._recognize_transformers_whisper(audio_path, model_name, language, prompt, max_sentence_length)

    def _recognize_faster_whisper(
        self,
        audio_path: str,
        model_name: str,
        language: str,
        prompt: Optional[str],
        max_sentence_length: int
    ) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """faster-whisper 识别"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("请安装 faster-whisper: pip install faster-whisper")

        # 检查模型路径
        if not self.model_path:
            raise ValueError("无法获取模型目录")

        model_full_path = os.path.join(self.model_path, model_name)

        # 检查模型是否存在，不存在则尝试自动下载
        if not os.path.exists(model_full_path):
            _log_warning(f"模型不存在: {model_full_path}")

            # 检查是否启用自动下载
            auto_download = True
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from config import AUTO_DOWNLOAD_MODELS
                auto_download = AUTO_DOWNLOAD_MODELS
            except ImportError:
                pass  # 如果配置文件不存在，默认启用自动下载

            if auto_download:
                _log_info(f"🚀 尝试自动下载模型...")

                # 尝试自动下载
                if _download_model(model_name, model_full_path):
                    _log_info(f"✅ 模型下载成功，继续加载...")
                else:
                    # 下载失败，抛出错误
                    raise FileNotFoundError(
                        f"模型文件不存在且自动下载失败: {model_full_path}\n"
                        f"请手动下载模型并放置到 ComfyUI/models/TTS/{model_name} 目录\n"
                        f"推荐模型:\n"
                        f"  - Belle-whisper-large-v3-zh-punct-ct2 (中文)\n"
                        f"  - Belle-whisper-large-v3-zh-punct-ct2-float32 (中文, float32)\n"
                        f"  - whisper-large-v3-ct2 (多语言)\n"
                        f"下载命令:\n"
                        f"  pip install huggingface-hub\n"
                        f"  huggingface-cli download <repo_id> --local-dir {model_full_path}"
                    )
            else:
                # 自动下载已禁用
                raise FileNotFoundError(
                    f"模型文件不存在: {model_full_path}\n"
                    f"自动下载已禁用（在 config.py 中设置 AUTO_DOWNLOAD_MODELS = True 启用）\n"
                    f"请手动下载模型并放置到 ComfyUI/models/TTS/{model_name} 目录\n"
                    f"推荐模型:\n"
                    f"  - Belle-whisper-large-v3-zh-punct-ct2 (中文)\n"
                    f"  - Belle-whisper-large-v3-zh-punct-ct2-float32 (中文, float32)\n"
                    f"  - whisper-large-v3-ct2 (多语言)\n"
                    f"下载命令:\n"
                    f"  pip install huggingface-hub\n"
                    f"  huggingface-cli download <repo_id> --local-dir {model_full_path}"
                )

        # 加载模型（使用缓存）
        if self.current_model_name != model_name:
            _log_info(f"📥 加载模型: {model_name}")
            _log_info(f"📁 模型路径: {model_full_path}")

            # 清空之前的模型
            self.model_cache.clear()

            # 加载新模型
            self.model_cache["current"] = WhisperModel(
                model_full_path,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.current_model_name = model_name
            _log_info(f"✅ 模型加载完成")
        else:
            _log_info(f"♻️ 使用缓存的模型: {model_name}")

        model = self.model_cache["current"]

        # faster-whisper 识别（支持 word_timestamps 和 prompt）
        _log_info("🎤 开始语音识别...")

        # 准备 transcribe 参数
        transcribe_kwargs = {
            "language": None if language == "auto" else language,
            "word_timestamps": True
        }

        # 如果提供了 prompt，添加到参数中（用于提高准确率）
        if prompt and prompt.strip():
            transcribe_kwargs["initial_prompt"] = prompt.strip()
            _log_info(f"💡 使用提示词提高准确率: {prompt.strip()[:50]}...")

        segments, info = model.transcribe(audio_path, **transcribe_kwargs)

        _log_info(f"🌍 检测到语言: {info.language}")

        # 提取逐词和逐句时间戳
        words_list = []
        sentences_list = []

        # 将 segments 转换为列表以便调试
        segments_list = list(segments)
        _log_info(f"🔍 模型返回了 {len(segments_list)} 个 segment")

        for i, segment in enumerate(segments_list):
            # 句子级别
            start = round(segment.start, 2)
            end = round(segment.end, 2)
            text = segment.text.strip()
            sentences_list.append((start, end, text))

            # 调试：显示前 3 个 segment 的信息
            if i < 3:
                _log_info(f"  Segment {i+1}: [{start:.2f}s - {end:.2f}s] {text[:50]}...")

            # 词级别
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    word_start = round(word.start, 2)
                    word_end = round(word.end, 2)
                    word_text = word.word.strip()
                    words_list.append((word_start, word_end, word_text))

        _log_info(f"📝 识别了 {len(sentences_list)} 个句子, {len(words_list)} 个词")

        # 如果只有1个句子且文本很长，强制分句
        if len(sentences_list) == 1 and len(sentences_list[0][2]) > 30:
            _log_info(f"🔧 检测到只有1个长句子，使用标点符号强制分句...")
            sentences_list = self._force_split_sentences(sentences_list)
            _log_info(f"✅ 强制分句后得到 {len(sentences_list)} 个句子")

        # 如果模型不支持逐词时间戳，使用 jieba 分词生成伪时间戳
        if not words_list and sentences_list:
            _log_info(f"🔧 模型不支持逐词时间戳，使用 jieba 分词生成伪时间戳...")
            words_list = self._generate_words_from_sentences(sentences_list)
            _log_info(f"✅ 生成了 {len(words_list)} 个词的伪时间戳")

        # 如果需要，按最大长度重新分句
        if max_sentence_length > 0:
            sentences_list = self._split_sentences(words_list, sentences_list, max_sentence_length, info.language)

        return words_list, sentences_list

    def _recognize_transformers_whisper(
        self,
        audio_path: str,
        model_name: str,
        language: str,
        prompt: Optional[str],
        max_sentence_length: int
    ) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
        """Transformers Whisper 识别（用于 Belle-whisper-large-v3-zh-punct 等模型）"""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        # 构建模型路径
        if not self.model_path:
            raise ValueError("无法获取模型目录")

        model_full_path = os.path.join(self.model_path, model_name)

        # 检查模型是否存在，不存在则尝试自动下载
        if not os.path.exists(model_full_path):
            _log_warning(f"模型不存在: {model_full_path}")

            # 检查是否启用自动下载
            auto_download = True
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from config import AUTO_DOWNLOAD_MODELS
                auto_download = AUTO_DOWNLOAD_MODELS
            except ImportError:
                pass  # 如果配置文件不存在，默认启用自动下载

            if auto_download:
                _log_info(f"🚀 尝试自动下载模型...")

                # 尝试自动下载
                if _download_model(model_name, model_full_path):
                    _log_info(f"✅ 模型下载成功，继续加载...")
                else:
                    # 下载失败，抛出错误
                    raise FileNotFoundError(
                        f"模型文件不存在且自动下载失败: {model_full_path}\n"
                        f"请手动下载模型并放置到 ComfyUI/models/TTS/{model_name} 目录\n"
                        f"下载命令:\n"
                        f"  pip install huggingface-hub\n"
                        f"  huggingface-cli download BELLE-2/{model_name} --local-dir {model_full_path}"
                    )
            else:
                # 自动下载已禁用
                raise FileNotFoundError(
                    f"模型文件不存在: {model_full_path}\n"
                    f"自动下载已禁用（在 config.py 中设置 AUTO_DOWNLOAD_MODELS = True 启用）\n"
                    f"请手动下载模型并放置到 ComfyUI/models/TTS/{model_name} 目录\n"
                    f"下载命令:\n"
                    f"  pip install huggingface-hub\n"
                    f"  huggingface-cli download BELLE-2/{model_name} --local-dir {model_full_path}"
                )

        # 加载或使用缓存的模型
        if self.current_model_name != model_name:
            _log_info(f"📥 加载 Transformers Whisper 模型: {model_name}...")
            self.model_cache.clear()

            # 创建 pipeline
            self.model_cache["current"] = pipeline(
                "automatic-speech-recognition",
                model=model_full_path,
                device=0 if self.device == "cuda" else -1
            )

            # 设置强制解码器 ID（语言和任务）
            self.model_cache["current"].model.config.forced_decoder_ids = (
                self.model_cache["current"].tokenizer.get_decoder_prompt_ids(
                    language="zh" if language == "auto" else language,
                    task="transcribe"
                )
            )

            self.current_model_name = model_name
            _log_info(f"✅ 模型加载完成")

        transcriber = self.model_cache["current"]

        # 执行识别（Transformers pipeline 返回句子级别的结果）
        _log_info(f"🎤 开始识别...")

        # 准备识别参数
        transcribe_kwargs = {
            "return_timestamps": True,  # 返回时间戳
            "chunk_length_s": 30,  # 30秒分块
            "stride_length_s": 5   # 5秒重叠
        }

        # 如果提供了 prompt，添加到 generate_kwargs 中（用于提高准确率）
        if prompt and prompt.strip():
            transcribe_kwargs["generate_kwargs"] = {"prompt_ids": transcriber.tokenizer.encode(prompt.strip())}
            _log_info(f"💡 使用提示词提高准确率: {prompt.strip()[:50]}...")

        result = transcriber(audio_path, **transcribe_kwargs)

        # 提取句子级别的时间戳
        sentences_list = []
        words_list = []

        if "chunks" in result:
            # 有时间戳信息
            _log_info(f"🔍 Transformers 返回了 {len(result['chunks'])} 个 chunk")
            for i, chunk in enumerate(result["chunks"]):
                start = round(chunk["timestamp"][0], 2) if chunk["timestamp"][0] is not None else 0.0
                end = round(chunk["timestamp"][1], 2) if chunk["timestamp"][1] is not None else 0.0
                text = chunk["text"].strip()
                if text:
                    sentences_list.append((start, end, text))
                    # 调试：显示前 3 个 chunk 的信息
                    if i < 3:
                        _log_info(f"  Chunk {i+1}: [{start:.2f}s - {end:.2f}s] {text[:50]}...")
        else:
            # 没有时间戳信息，使用整个文本
            _log_warning(f"⚠️ Transformers 没有返回 chunks，所有文本将作为一个句子")
            text = result["text"].strip()
            if text:
                sentences_list.append((0.0, 0.0, text))

        _log_info(f"📝 识别了 {len(sentences_list)} 个句子")

        # 如果只有1个句子且文本很长，强制分句
        if len(sentences_list) == 1 and len(sentences_list[0][2]) > 30:
            _log_info(f"🔧 检测到只有1个长句子，使用标点符号强制分句...")
            sentences_list = self._force_split_sentences(sentences_list)
            _log_info(f"✅ 强制分句后得到 {len(sentences_list)} 个句子")

        # Transformers pipeline 不支持逐词时间戳
        # 从句子中生成伪逐词时间戳（使用 jieba 分词）
        if not words_list and sentences_list:
            _log_info(f"🔧 Transformers 模型不支持逐词时间戳，使用 jieba 分词生成伪时间戳...")
            words_list = self._generate_words_from_sentences(sentences_list)
            _log_info(f"✅ 生成了 {len(words_list)} 个词的伪时间戳")

        return words_list, sentences_list

    def _force_split_sentences(
        self,
        sentences_list: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        """
        强制分句（使用标点符号）

        当模型只返回1个长句子时，使用标点符号强制分句

        Args:
            sentences_list: 句子列表 [(start, end, text), ...]

        Returns:
            new_sentences_list: 分句后的句子列表
        """
        import re

        if len(sentences_list) != 1:
            return sentences_list

        start, end, text = sentences_list[0]

        # 如果文本太短，不需要分句
        if len(text) < 30:
            return sentences_list

        # 使用标点符号分句（中文和英文标点）
        # 匹配句子结束符号：。！？.!?
        sentences = re.split(r'([。！？.!?]+)', text)

        # 重新组合句子（保留标点符号）
        split_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]

            sentence = sentence.strip()
            if sentence:
                split_sentences.append(sentence)

        # 如果最后一个元素没有标点符号，也添加进去
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            split_sentences.append(sentences[-1].strip())

        # 如果分句失败（只有1个句子），尝试使用逗号分句
        if len(split_sentences) <= 1:
            sentences = re.split(r'([，,]+)', text)
            split_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    sentence = sentences[i] + sentences[i + 1]
                else:
                    sentence = sentences[i]

                sentence = sentence.strip()
                if sentence:
                    split_sentences.append(sentence)

            if len(sentences) % 2 == 1 and sentences[-1].strip():
                split_sentences.append(sentences[-1].strip())

        # 如果还是只有1个句子，返回原始列表
        if len(split_sentences) <= 1:
            return sentences_list

        # 计算每个句子的时长（根据字符数按比例分配）
        total_duration = end - start
        total_chars = sum(len(s) for s in split_sentences)

        # 生成新的 sentences_list
        new_sentences_list = []
        current_time = start
        for sentence in split_sentences:
            sentence_start = current_time
            # 根据句子长度按比例分配时间
            sentence_duration = (len(sentence) / total_chars) * total_duration
            sentence_end = current_time + sentence_duration
            new_sentences_list.append((round(sentence_start, 2), round(sentence_end, 2), sentence))
            current_time = sentence_end

        return new_sentences_list

    def _generate_words_from_sentences(
        self,
        sentences_list: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        """
        从句子列表生成伪逐词时间戳（使用 jieba 分词）

        Args:
            sentences_list: 句子列表 [(start, end, text), ...]

        Returns:
            words_list: 词列表 [(start, end, word), ...]
        """
        import jieba

        words_list = []

        for idx, (sentence_start, sentence_end, sentence_text) in enumerate(sentences_list):
            # 使用 jieba 分词（保留空格）
            words = [w for w in jieba.cut(sentence_text) if w]  # 只过滤空字符串，保留空格

            if not words:
                continue

            # 计算每个词的平均时长
            sentence_duration = sentence_end - sentence_start
            word_duration = sentence_duration / len(words)

            # 为每个词分配时间戳
            current_time = sentence_start
            for word_idx, word in enumerate(words):
                word_start = current_time
                word_end = current_time + word_duration
                words_list.append((round(word_start, 2), round(word_end, 2), word))
                current_time = word_end

            # 在每个句子结束后添加换行符标记（除了最后一个句子）
            if idx < len(sentences_list) - 1:
                # 添加一个特殊的换行符标记，时间戳与最后一个词相同
                # 使用 <NEWLINE> 标记而不是 \n，因为 \n 会被 strip() 去掉
                words_list.append((round(word_end, 2), round(word_end, 2), "<NEWLINE>"))

        return words_list

    def _split_sentences(
        self,
        words_list: List[Tuple[float, float, str]],
        sentences_list: List[Tuple[float, float, str]],
        max_length: int,
        language: str
    ) -> List[Tuple[float, float, str]]:
        """按最大长度重新分句（参考 ComfyUI_ASR 的实现）"""
        # 这里可以实现更复杂的分句逻辑
        # 暂时返回原始句子列表
        return sentences_list

    def unload_model(self):
        """卸载当前模型"""
        self.model_cache.clear()
        self.current_model_name = None
        torch.cuda.empty_cache()
        _log_info(f"🗑️ 已卸载所有模型")

