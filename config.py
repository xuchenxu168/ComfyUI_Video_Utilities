"""
ComfyUI Video Utilities 配置文件
"""

# ========================================
# 模型自动下载配置
# ========================================

# 是否启用模型自动下载
# True: 当模型不存在时，自动从 Hugging Face 下载
# False: 当模型不存在时，抛出错误，需要手动下载
AUTO_DOWNLOAD_MODELS = True

# 模型下载超时时间（秒）
# 大模型下载可能需要较长时间，建议设置为 3600（1小时）或更长
DOWNLOAD_TIMEOUT = 3600

# 是否使用 Hugging Face 镜像站点
# True: 使用镜像站点（适合国内用户）
# False: 使用官方站点
USE_HF_MIRROR = False

# Hugging Face 镜像站点地址
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# ========================================
# ASR 配置
# ========================================

# 默认 ASR 引擎
# "faster-whisper": 使用 CTranslate2 格式模型（推荐）
# "transformers": 使用 Transformers 格式模型（需要 torchcodec）
DEFAULT_ASR_ENGINE = "faster-whisper"

# 默认语言
# "auto": 自动检测
# "zh": 中文
# "en": 英文
DEFAULT_LANGUAGE = "auto"

# 默认最大句子长度
DEFAULT_MAX_SENTENCE_LENGTH = 50

# ========================================
# 字幕渲染配置
# ========================================

# 默认字体
DEFAULT_FONT = "SourceHanSansCN-Bold.otf"

# 默认字体大小
DEFAULT_FONT_SIZE = 48

# 默认字体颜色
DEFAULT_FONT_COLOR = "#FFFF00"

# 默认描边宽度
DEFAULT_STROKE_WIDTH = 2

# 默认描边颜色
DEFAULT_STROKE_COLOR = "#000000"

# 默认文本方向
# "horizontal": 横向
# "vertical": 纵向
DEFAULT_TEXT_DIRECTION = "horizontal"

# 默认动画类型
# "none", "fade_in", "slide_up", "slide_down", "slide_left", "slide_right",
# "zoom_in", "zoom_out", "bounce", "typewriter", "wave", "shake"
DEFAULT_ANIMATION = "fade_in"

# 默认动画持续时间（秒）
DEFAULT_ANIMATION_DURATION = 0.3

# ========================================
# 视频处理配置
# ========================================

# 默认视频编码器
# "libx264": H.264 编码（推荐，兼容性好）
# "mpeg4": MPEG-4 编码（文件更小）
DEFAULT_VIDEO_CODEC = "libx264"

# 默认视频质量（CRF）
# 0-51，数值越小质量越高，文件越大
# 推荐值: 18-28
DEFAULT_VIDEO_CRF = 23

# 默认音频编码器
DEFAULT_AUDIO_CODEC = "aac"

# 默认音频比特率
DEFAULT_AUDIO_BITRATE = "128k"

# ========================================
# 调试配置
# ========================================

# 是否启用调试日志
DEBUG = False

# 是否保留临时文件
KEEP_TEMP_FILES = False

