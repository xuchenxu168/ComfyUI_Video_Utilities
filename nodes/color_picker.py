"""
Color Picker Node
颜色选择器节点
"""


class VideoUtilitiesColorPicker:
    """颜色选择器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": "#FFFF00"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("颜色代码",)
    FUNCTION = "pick_color"
    CATEGORY = "Video_Utilities/Utility"
    
    def pick_color(self, color):
        """
        返回颜色代码
        
        Args:
            color: 十六进制颜色代码
        
        Returns:
            颜色代码字符串
        """
        # 确保颜色代码格式正确
        if not color.startswith('#'):
            color = '#' + color
        
        # 验证颜色代码
        color = color.upper()
        if len(color) != 7:
            print(f"[ColorPicker] 警告: 颜色代码格式不正确: {color}，使用默认黄色")
            color = "#FFFF00"
        
        return (color,)

