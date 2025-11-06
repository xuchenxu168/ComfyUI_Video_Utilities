"""
智能文本换行
支持中文（jieba 分词）和英文（按单词）
"""

import re

# 标点符号定义
PUNCTUATION = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃『』【】〖〗〘〙〚〛〜〝〞〟–—''‛„‟…‧﹏." \
              "!?(),;:[]{}<>\"+-=&^*%$#@/" \
              "。？！，、；：""''《》〈〉「」〔〕——·~`-"


def smart_wrap_text(text: str, max_width: int, font_path: str, font_size: int, language: str = 'zh') -> str:
    """
    智能换行
    
    Args:
        text: 要换行的文本
        max_width: 最大宽度（像素）
        font_path: 字体路径
        font_size: 字体大小
        language: 语言（zh/en）
    
    Returns:
        换行后的文本
    """
    if not text.strip():
        return ""
    
    try:
        # 尝试使用 moviepy 的 TextClip 计算宽度
        from moviepy.video.VideoClip import TextClip
        
        if language == 'zh':
            return _wrap_chinese(text, max_width, font_path, font_size)
        else:
            return _wrap_english(text, max_width, font_path, font_size)
    except Exception as e:
        print(f"[TextWrapper] 智能换行失败，使用简单换行: {e}")
        # 回退到简单换行
        return _simple_wrap(text, max_width, font_size, language)


def _wrap_chinese(text: str, max_width: int, font_path: str, font_size: int) -> str:
    """中文换行（使用 jieba 分词）"""
    try:
        import jieba
        from PIL import ImageFont, ImageDraw, Image
    except ImportError:
        print("[TextWrapper] jieba 或 PIL 未安装，使用简单换行")
        return _simple_wrap(text, max_width, font_size, 'zh')

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"[TextWrapper] 无法加载字体 {font_path}，使用简单换行")
        return _simple_wrap(text, max_width, font_size, 'zh')

    # 创建临时 draw 对象用于测量文本
    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)

    # 使用 jieba 分词（保留空格）
    words = [w for w in jieba.cut(text) if w]  # 只过滤空字符串，保留空格

    lines = []
    current_line = ""

    for word in words:
        # 测试添加这个词后的宽度
        test_line = current_line + word

        # 使用 PIL 计算文本宽度
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            # 当前行已满，开始新行
            if current_line:
                lines.append(current_line)
            current_line = word

    # 添加最后一行
    if current_line:
        lines.append(current_line)

    # 处理标点符号：将行首的标点移到上一行末尾
    corrected_lines = []
    for i, line in enumerate(lines):
        if i > 0 and line and line[0] in PUNCTUATION:
            # 将标点移到上一行
            if corrected_lines:
                corrected_lines[-1] += line[0]
                line = line[1:]
        if line.strip():
            corrected_lines.append(line)

    return '\n'.join(corrected_lines)


def _wrap_english(text: str, max_width: int, font_path: str, font_size: int) -> str:
    """英文换行（按单词）"""
    try:
        from moviepy.video.VideoClip import TextClip
    except ImportError:
        print("[TextWrapper] moviepy 未安装，使用简单换行")
        return _simple_wrap(text, max_width, font_size, 'en')
    
    words = text.split(' ')
    
    lines = []
    current_line_words = []
    
    for word in words:
        if not word:
            continue
        
        # 测试添加这个词后的宽度
        temp_line_words = current_line_words + [word]
        test_line = " ".join(temp_line_words)
        
        try:
            test_clip = TextClip(text=test_line, font=font_path, font_size=font_size, method='label')
            line_width = test_clip.w
            test_clip.close()
        except:
            # 如果创建失败，估算宽度
            line_width = len(test_line) * font_size * 0.6
        
        if line_width <= max_width:
            current_line_words.append(word)
        else:
            # 当前行已满
            # 检查是否需要将标点移到上一行
            if word and word[0] in PUNCTUATION and len(current_line_words) > 0:
                # 将上一个词移到新行
                word_to_move = current_line_words.pop()
                lines.append(" ".join(current_line_words))
                current_line_words = [word_to_move, word]
            else:
                lines.append(" ".join(current_line_words))
                current_line_words = [word]
    
    # 添加最后一行
    if current_line_words:
        lines.append(" ".join(current_line_words))
    
    return '\n'.join([line for line in lines if line.strip()])


def _simple_wrap(text: str, max_width: int, font_size: int, language: str) -> str:
    """简单换行（回退方案）"""
    # 估算每行字符数
    if language == 'zh':
        chars_per_line = int(max_width / (font_size * 0.9))
    else:
        chars_per_line = int(max_width / (font_size * 0.6))
    
    if chars_per_line <= 0:
        chars_per_line = 10
    
    lines = []
    current_line = ""
    
    if language == 'zh':
        # 中文：按字符换行
        for char in text:
            if len(current_line) < chars_per_line:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
    else:
        # 英文：按单词换行
        words = text.split(' ')
        for word in words:
            if len(current_line) + len(word) + 1 <= chars_per_line:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return '\n'.join(lines)


def remove_punctuation(text: str, language: str = 'en') -> str:
    """
    去除标点符号
    
    Args:
        text: 原始文本
        language: 语言（zh/en）
    
    Returns:
        去除标点后的文本
    """
    if language == 'en':
        # 英文：保留缩写中的撇号和连字符
        APOS_PLACEHOLDER = '¤'
        HYPHEN_PLACEHOLDER = '§'
        
        # 保护缩写（don't, it's）
        processed_text = re.sub(r"(?<=[a-zA-Z])['''](?=[a-zA-Z])", APOS_PLACEHOLDER, text)
        # 保护连字符（well-known）
        processed_text = re.sub(r"(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])", HYPHEN_PLACEHOLDER, processed_text)
        
        # 移除所有标点
        punct_pattern = re.compile(f"[{re.escape(PUNCTUATION)}]+")
        processed_text = punct_pattern.sub(" ", processed_text)
        
        # 恢复保护的字符
        processed_text = processed_text.replace(APOS_PLACEHOLDER, "'")
        processed_text = processed_text.replace(HYPHEN_PLACEHOLDER, "-")
    else:
        # 中文：直接移除所有标点
        punct_pattern = re.compile(f"[{re.escape(PUNCTUATION)}]+")
        processed_text = punct_pattern.sub(" ", text)
    
    # 合并多个空格
    cleaned_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return cleaned_text

