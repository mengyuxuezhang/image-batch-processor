"""
图片处理模块
"""

from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageOps, ImageFilter
import os
import logging
import hashlib
import random
import json
import numpy as np
import cv2
from datetime import datetime

# 创建本地日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageProcessor:
    """图片处理器类"""
    
    def __init__(self):
        """初始化图片处理器"""
        pass
    
    def process_images(self, image_paths, output_folder=None, options=None):
        """
        批量处理图片
        
        参数:
        - image_paths: 图片路径列表
        - output_folder: 输出文件夹路径（可选，默认在原文件夹中创建processed子文件夹）
        - options: 处理选项字典，包括:
            - watermark_text: 水印文字内容
            - brightness_factor: 亮度调节因子
            - contrast_factor: 对比度调节因子
            - saturation_factor: 饱和度调节因子
            - sharpness_factor: 锐化调节因子
            - blur_radius: 模糊半径
            - resize_factor: 尺寸调整因子
            - crop_percentage: 裁剪百分比（上下左右各裁掉的百分比）
            - add_border: 是否添加边框
            - add_noise: 是否添加噪点
            - remove_metadata: 是否删除元数据
            - change_md5: 是否修改MD5值
            
        返回:
        - processed_paths: 处理后的图片路径列表
        """
        if options is None:
            options = {
                'watermark_text': "Watermark",
                'brightness_factor': 1.1,
                'contrast_factor': 1.0,
                'saturation_factor': 1.0,
                'sharpness_factor': 1.0,
                'blur_radius': 0.0,
                'resize_factor': 1.0,  # 默认不调整尺寸
                'crop_percentage': 0.0,  # 默认不裁剪
                'add_border': False,
                'add_noise': False,
                'remove_metadata': True,
                'change_md5': False
            }
        
        # 如果没有指定输出文件夹，在原文件夹中创建processed子文件夹
        if output_folder is None and image_paths:
            first_image_dir = os.path.dirname(image_paths[0])
            output_folder = os.path.join(first_image_dir, "processed")
        
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"创建输出文件夹: {output_folder}")
        
        processed_paths = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"正在处理图片 {i+1}/{len(image_paths)}: {image_path}")
                
                # 生成输出路径
                file_name = os.path.basename(image_path)
                file_name_without_ext, file_ext = os.path.splitext(file_name)
                
                # 如果需要修改MD5，在文件名中添加随机数
                if options.get('change_md5', False):
                    file_name_without_ext += f"_{random.randint(1000, 9999)}"
                
                output_path = os.path.join(output_folder, f"{file_name_without_ext}_processed{file_ext}")
                
                # 处理单张图片
                result_path = self.process_single_image(
                    image_path, 
                    output_path, 
                    options
                )
                
                if result_path:
                    processed_paths.append(result_path)
                    logger.info(f"✅ 图片处理完成: {result_path}")
                else:
                    logger.error(f"❌ 图片处理失败: {image_path}")
                    
            except Exception as e:
                logger.error(f"处理图片时出错 {image_path}: {str(e)}")
        
        logger.info(f"图片处理完成，共处理 {len(processed_paths)} 张图片")
        return processed_paths
    
    def get_image_files(self, folder_path):
        """
        获取文件夹中的所有图片文件
        
        参数:
        - folder_path: 文件夹路径
        
        返回:
        - 图片文件路径列表
        """
        if not os.path.exists(folder_path):
            logger.error(f"文件夹不存在: {folder_path}")
            return []
            
        # 支持的图片格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        
        # 获取所有图片文件
        image_files = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, file_name))
                
        logger.info(f"在文件夹 {folder_path} 中找到 {len(image_files)} 个图片文件")
        return image_files
    
    def get_chinese_font(self, size=24):
        """
        获取中文字体
        尝试几种常见的中文字体路径
        """
        # Windows系统常见的中文字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",    # 黑体
            "C:/Windows/Fonts/simsun.ttc",    # 宋体
            "C:/Windows/Fonts/msyhui.ttc",    # 微软雅黑UI
        ]
        
        # 尝试加载系统中文字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    from PIL import ImageFont
                    return ImageFont.truetype(font_path, size)
                except Exception as e:
                    logger.warning(f"加载字体 {font_path} 失败: {str(e)}")
                    continue
        
        # 如果找不到系统字体，尝试使用默认字体
        logger.warning("未能加载系统中文字体，使用默认字体")
        try:
            # 尝试使用默认字体
            return ImageFont.load_default()
        except:
            # 如果所有方法都失败，创建一个简单的字体
            return ImageFont.load_default()
    
    def add_noise(self, image, noise_factor=0.1):
        """
        为图片添加随机噪点
        """
        # 将PIL图像转换为numpy数组
        img_array = np.array(image)
        
        # 生成随机噪声
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        
        # 添加噪声到图像
        noisy_img = img_array + noise
        
        # 确保像素值在有效范围内
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # 转换回PIL图像
        return Image.fromarray(noisy_img)
    
    def adjust_hue(self, image, hue_factor=0.1):
        """
        调整图片色相
        hue_factor: 色相调整因子，范围-1到1
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 转换为HSV色彩空间
        if img_array.shape[2] == 3:  # RGB
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 调整色相
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
            
            # 转换回RGB
            img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(img_array)
    
    def adjust_color_temperature(self, image, temp_factor=0.1):
        """
        调整图片色温
        temp_factor: 色温调整因子，正值偏暖，负值偏冷，范围-1到1
        """
        img_array = np.array(image)
        
        # 创建色温调整矩阵
        if temp_factor > 0:  # 偏暖
            matrix = np.array([
                [1 + temp_factor * 0.3, 0, 0],
                [0, 1, 0],
                [0, 0, 1 - temp_factor * 0.1]
            ])
        else:  # 偏冷
            temp_factor = abs(temp_factor)
            matrix = np.array([
                [1 - temp_factor * 0.1, 0, 0],
                [0, 1, 0],
                [0, 0, 1 + temp_factor * 0.3]
            ])
        
        # 应用矩阵
        if img_array.shape[2] == 3:  # RGB
            result = np.zeros_like(img_array)
            for i in range(3):
                result[:, :, i] = img_array[:, :, 0] * matrix[i, 0] + \
                                  img_array[:, :, 1] * matrix[i, 1] + \
                                  img_array[:, :, 2] * matrix[i, 2]
            
            # 确保像素值在有效范围内
            result = np.clip(result, 0, 255).astype(np.uint8)
            return Image.fromarray(result)
        
        return image
    
    def adjust_color_balance(self, image, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
        """
        调整色彩平衡
        red_factor, green_factor, blue_factor: RGB三色调整因子
        """
        img_array = np.array(image)
        
        if img_array.shape[2] == 3:  # RGB
            # 分别调整RGB通道
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_factor, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * green_factor, 0, 255)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_factor, 0, 255)
            
            return Image.fromarray(img_array)
        
        return image
    
    def convert_to_grayscale(self, image, method='weighted'):
        """
        将彩色图片转换为黑白
        method: 转换方法，'weighted'(加权平均), 'average'(简单平均), 'luminosity'(亮度法)
        """
        img_array = np.array(image)
        
        if img_array.shape[2] == 3:  # RGB
            if method == 'weighted':
                # 加权平均法 (0.299*R + 0.587*G + 0.114*B)
                gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            elif method == 'average':
                # 简单平均法
                gray = np.mean(img_array, axis=2)
            else:  # luminosity
                # 亮度法 (0.21*R + 0.72*G + 0.07*B)
                gray = 0.21 * img_array[:, :, 0] + 0.72 * img_array[:, :, 1] + 0.07 * img_array[:, :, 2]
            
            # 转换为3通道灰度图
            gray_3d = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
            return Image.fromarray(gray_3d)
        
        return image
    
    def adjust_shadows_highlights(self, image, shadow_factor=0.2, highlight_factor=0.2):
        """
        调整阴影和高光
        shadow_factor: 阴影调整因子，0-1，值越大阴影越亮
        highlight_factor: 高光调整因子，0-1，值越大高光越暗
        """
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        if img_array.shape[2] == 3:  # RGB
            # 转换为LAB色彩空间
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # 调整亮度通道
            l = lab[:, :, 0]
            
            # 调整阴影（暗部）
            shadow_mask = 1.0 - l
            l = l + shadow_mask * shadow_factor
            
            # 调整高光（亮部）
            highlight_mask = l
            l = l - highlight_mask * highlight_factor
            
            # 确保值在有效范围内
            l = np.clip(l, 0, 1)
            
            # 更新LAB图像
            lab[:, :, 0] = l
            
            # 转换回RGB
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
        
        return image
    
    def adjust_local_brightness(self, image, center_x=0.5, center_y=0.5, radius=0.3, brightness_factor=1.2):
        """
        局部亮度调整
        center_x, center_y: 调整中心位置，范围0-1
        radius: 调整半径，范围0-1
        brightness_factor: 亮度调整因子
        """
        img_array = np.array(image, dtype=np.float32)
        
        height, width = img_array.shape[:2]
        
        # 创建径向渐变掩码
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - width * center_x) ** 2 + (y - height * center_y) ** 2)
        max_dist = np.sqrt(width ** 2 + height ** 2) / 2
        
        # 创建平滑渐变
        mask = np.exp(-(dist_from_center / (max_dist * radius)) ** 2)
        
        # 应用亮度调整
        if len(img_array.shape) == 3:  # 彩色图像
            for i in range(3):
                img_array[:, :, i] = img_array[:, :, i] * (1 + mask * (brightness_factor - 1))
        else:  # 灰度图像
            img_array = img_array * (1 + mask * (brightness_factor - 1))
        
        # 确保像素值在有效范围内
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def apply_gradient_map(self, image, gradient_type='linear', direction=0, intensity=0.3):
        """
        应用渐变映射
        gradient_type: 渐变类型，'linear'(线性), 'radial'(径向)
        direction: 渐变方向，0-7（0=右，1=下，2=左，3=上，4-7=对角线）
        intensity: 渐变强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # 创建渐变掩码
        if gradient_type == 'linear':
            if direction == 0:  # 右
                gradient = np.linspace(0, 1, width).reshape(1, -1)
                gradient = np.repeat(gradient, height, axis=0)
            elif direction == 1:  # 下
                gradient = np.linspace(0, 1, height).reshape(-1, 1)
                gradient = np.repeat(gradient, width, axis=1)
            elif direction == 2:  # 左
                gradient = np.linspace(1, 0, width).reshape(1, -1)
                gradient = np.repeat(gradient, height, axis=0)
            elif direction == 3:  # 上
                gradient = np.linspace(1, 0, height).reshape(-1, 1)
                gradient = np.repeat(gradient, width, axis=1)
            elif direction == 4:  # 右下对角线
                y, x = np.ogrid[:height, :width]
                gradient = (x + y) / (width + height)
            elif direction == 5:  # 左下对角线
                y, x = np.ogrid[:height, :width]
                gradient = (width - x + y) / (width + height)
            elif direction == 6:  # 左上对角线
                y, x = np.ogrid[:height, :width]
                gradient = (width - x + height - y) / (width + height)
            else:  # 右上对角线
                y, x = np.ogrid[:height, :width]
                gradient = (x + height - y) / (width + height)
        else:  # radial
            y, x = np.ogrid[:height, :width]
            center_x, center_y = width // 2, height // 2
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            gradient = dist_from_center / max_dist
        
        # 应用渐变
        if len(img_array.shape) == 3:  # 彩色图像
            for i in range(3):
                img_array[:, :, i] = img_array[:, :, i] * (1 + gradient * intensity)
        else:  # 灰度图像
            img_array = img_array * (1 + gradient * intensity)
        
        # 确保像素值在有效范围内
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def adjust_exposure(self, image, exposure_factor=0.2):
        """
        调整图片曝光度
        exposure_factor: 曝光调整因子，-1到1，正值增加曝光，负值减少曝光
        """
        img_array = np.array(image, dtype=np.float32)
        
        # 应用曝光调整
        img_array = img_array * (1 + exposure_factor)
        
        # 确保像素值在有效范围内
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def smart_sharpen(self, image, amount=1.0, radius=1.0, threshold=0):
        """
        智能锐化
        amount: 锐化强度，0-2
        radius: 锐化半径，0.1-5
        threshold: 锐化阈值，0-255
        """
        img_array = np.array(image, dtype=np.float32)
        
        # 创建高斯模糊版本
        blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
        
        # 计算差异
        diff = img_array - blurred
        
        # 应用阈值
        if threshold > 0:
            mask = np.abs(diff) >= threshold
            diff = diff * mask
        
        # 应用锐化
        sharpened = img_array + diff * amount
        
        # 确保像素值在有效范围内
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sharpened)
    
    def reduce_noise(self, image, strength=0.5):
        """
        减少图片噪点
        strength: 降噪强度，0-1
        """
        img_array = np.array(image)
        
        # 使用双边滤波进行降噪
        if len(img_array.shape) == 3:  # 彩色图像
            # 根据强度调整滤波参数
            d = int(9 * strength)
            sigma_color = 75 * strength
            sigma_space = 75 * strength
            
            denoised = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        else:  # 灰度图像
            d = int(9 * strength)
            sigma = 75 * strength
            
            denoised = cv2.bilateralFilter(img_array, d, sigma)
        
        return Image.fromarray(denoised)
    
    def add_grain_effect(self, image, grain_size=0.5, grain_strength=0.1):
        """
        添加胶片颗粒效果
        grain_size: 颗粒大小，0.1-2
        grain_strength: 颗粒强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        
        height, width = img_array.shape[:2]
        
        # 生成随机噪声
        noise = np.random.normal(0, grain_strength * 255, img_array.shape)
        
        # 调整颗粒大小
        if grain_size > 1:
            # 对噪声进行模糊以创建更大的颗粒
            ksize = int(grain_size * 2) + 1
            noise = cv2.GaussianBlur(noise, (ksize, ksize), 0)
        
        # 添加噪声到图像
        grainy_img = img_array + noise
        
        # 确保像素值在有效范围内
        grainy_img = np.clip(grainy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(grainy_img)
    
    def enhance_details(self, image, strength=0.5):
        """
        增强图片细节
        strength: 增强强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        
        # 使用Unsharp Mask增强细节
        blurred = cv2.GaussianBlur(img_array, (0, 0), 2.0)
        enhanced = cv2.addWeighted(img_array, 1 + strength, blurred, -strength, 0)
        
        # 确保像素值在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def slight_rotation(self, image, angle_range=5):
        """
        轻微旋转图片
        angle_range: 旋转角度范围，正负多少度
        """
        # 随机生成旋转角度
        angle = random.uniform(-angle_range, angle_range)
        
        # 旋转图片
        rotated = image.rotate(angle, expand=False, fillcolor='white')
        
        return rotated
    
    def adjust_perspective(self, image, distortion_factor=0.1):
        """
        调整图片透视效果
        distortion_factor: 扭曲因子，0-0.5
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 计算四个角的新位置
        offset_x = width * distortion_factor
        offset_y = height * distortion_factor
        
        # 随机决定哪个角移动
        corners = [
            [0, 0],  # 左上
            [width, 0],  # 右上
            [width, height],  # 右下
            [0, height]  # 左下
        ]
        
        # 随机选择一个角进行移动
        corner_idx = random.randint(0, 3)
        if corner_idx == 0:  # 左上
            corners[0] = [offset_x, offset_y]
        elif corner_idx == 1:  # 右上
            corners[1] = [width - offset_x, offset_y]
        elif corner_idx == 2:  # 右下
            corners[2] = [width - offset_x, height - offset_y]
        else:  # 左下
            corners[3] = [offset_x, height - offset_y]
        
        # 计算透视变换矩阵
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = np.float32(corners)
        
        # 应用透视变换
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img_array, matrix, (width, height))
        
        return Image.fromarray(warped)
    
    def slight_distortion(self, image, distortion_type='wave', strength=0.1):
        """
        轻微扭曲图片
        distortion_type: 扭曲类型，'wave'(波浪), 'ripple'(涟漪)
        strength: 扭曲强度，0-1
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        if distortion_type == 'wave':
            # 波浪扭曲
            x_distorted = x + strength * 10 * np.sin(2 * np.pi * y / (height / 2))
            y_distorted = y
        else:  # ripple
            # 涟漪扭曲
            center_x, center_y = width // 2, height // 2
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            x_distorted = x + strength * 5 * np.sin(dist / 10)
            y_distorted = y + strength * 5 * np.cos(dist / 10)
        
        # 确保坐标在有效范围内
        x_distorted = np.clip(x_distorted, 0, width - 1).astype(np.int32)
        y_distorted = np.clip(y_distorted, 0, height - 1).astype(np.int32)
        
        # 应用扭曲
        if len(img_array.shape) == 3:  # 彩色图像
            distorted = np.zeros_like(img_array)
            for i in range(3):
                distorted[:, :, i] = img_array[y_distorted, x_distorted, i]
        else:  # 灰度图像
            distorted = img_array[y_distorted, x_distorted]
        
        return Image.fromarray(distorted)
    
    def local_deformation(self, image, center_x=0.5, center_y=0.5, radius=0.2, strength=0.1):
        """
        局部变形
        center_x, center_y: 变形中心位置，范围0-1
        radius: 变形半径，范围0-1
        strength: 变形强度，-1到1，正值膨胀，负值收缩
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # 计算到中心的距离
        cx, cy = int(width * center_x), int(height * center_y)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(width ** 2 + height ** 2) / 2
        r = radius * max_dist
        
        # 创建变形掩码
        mask = np.exp(-(dist / r) ** 2)
        
        # 计算变形向量
        dx = (x - cx) / (dist + 1e-6) * mask * strength * r
        dy = (y - cy) / (dist + 1e-6) * mask * strength * r
        
        # 应用变形
        x_distorted = x + dx
        y_distorted = y + dy
        
        # 确保坐标在有效范围内
        x_distorted = np.clip(x_distorted, 0, width - 1).astype(np.int32)
        y_distorted = np.clip(y_distorted, 0, height - 1).astype(np.int32)
        
        # 应用变形
        if len(img_array.shape) == 3:  # 彩色图像
            deformed = np.zeros_like(img_array)
            for i in range(3):
                deformed[:, :, i] = img_array[y_distorted, x_distorted, i]
        else:  # 灰度图像
            deformed = img_array[y_distorted, x_distorted]
        
        return Image.fromarray(deformed)
    
    def add_vignette(self, image, vignette_strength=0.5):
        """
        添加晕影效果
        vignette_strength: 晕影强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # 创建径向渐变掩码
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        
        # 创建晕影掩码
        vignette = 1 - (dist_from_center / max_dist) * vignette_strength
        vignette = np.clip(vignette, 0, 1)
        
        # 应用晕影
        if len(img_array.shape) == 3:  # 彩色图像
            for i in range(3):
                img_array[:, :, i] = img_array[:, :, i] * vignette
        else:  # 灰度图像
            img_array = img_array * vignette
        
        # 确保像素值在有效范围内
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_chromatic_aberration(self, image, aberration_strength=0.5):
        """
        添加色差效果
        aberration_strength: 色差强度，0-5像素
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        if len(img_array.shape) == 3:  # 彩色图像
            # 创建色差效果
            r_channel = img_array[:, :, 0].copy()
            g_channel = img_array[:, :, 1].copy()
            b_channel = img_array[:, :, 2].copy()
            
            # 红通道向右偏移
            shift_r = int(aberration_strength)
            if shift_r > 0:
                r_channel[:, shift_r:] = r_channel[:, :-shift_r]
            
            # 蓝通道向左偏移
            shift_b = int(aberration_strength)
            if shift_b > 0:
                b_channel[:, :-shift_b] = b_channel[:, shift_b:]
            
            # 合并通道
            aberrated = np.stack([r_channel, g_channel, b_channel], axis=2)
            
            return Image.fromarray(aberrated)
        
        return image
    
    def add_soft_glow(self, image, glow_strength=0.3):
        """
        添加柔光效果
        glow_strength: 柔光强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        
        # 创建模糊版本
        blurred = cv2.GaussianBlur(img_array, (0, 0), 10)
        
        # 混合原图和模糊图
        glowing = (1 - glow_strength) * img_array + glow_strength * blurred
        
        # 确保像素值在有效范围内
        glowing = np.clip(glowing, 0, 255).astype(np.uint8)
        
        return Image.fromarray(glowing)
    
    def add_texture_overlay(self, image, texture_type='noise', intensity=0.1):
        """
        添加纹理叠加效果
        texture_type: 纹理类型，'noise'(噪点), 'film'(胶片), 'canvas'(画布)
        intensity: 纹理强度，0-1
        """
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # 创建纹理
        if texture_type == 'noise':
            # 随机噪点纹理
            texture = np.random.normal(0, 1, img_array.shape)
        elif texture_type == 'film':
            # 胶片颗粒纹理
            texture = np.random.normal(0, 1, img_array.shape)
            # 应用高斯模糊使颗粒更大
            texture = cv2.GaussianBlur(texture, (5, 5), 0)
        else:  # canvas
            # 画布纹理
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            texture = np.sin(x / 5) * np.cos(y / 5) * 10
        
        # 应用纹理
        textured = img_array + texture * intensity * 255
        
        # 确保像素值在有效范围内
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        
        return Image.fromarray(textured)
    
    def modify_capture_date(self, image, days_range=365):
        """
        修改拍摄日期
        days_range: 日期变化范围，正负多少天
        """
        # 这个函数需要修改EXIF数据，但由于我们已经在process_single_image中删除了元数据
        # 所以这个功能在这里不起作用
        # 如果需要保留元数据并修改，需要调整process_single_image中的remove_metadata选项
        return image
    
    def modify_camera_model(self, image):
        """
        修改相机型号
        """
        # 同样，这个功能需要保留元数据
        return image
    
    def modify_gps_info(self, image):
        """
        修改GPS信息
        """
        # 同样，这个功能需要保留元数据
        return image
    
    def modify_capture_params(self, image):
        """
        修改拍摄参数
        """
        # 同样，这个功能需要保留元数据
        return image
    
    def process_images_from_json(self, json_file_path, image_folder_path):
        """
        从JSON配置文件处理图片，并覆盖原图
        
        参数:
        - json_file_path: JSON配置文件的路径
        - image_folder_path: 图片文件夹路径
        
        返回:
        - processed_count: 成功处理的图片数量
        - total_count: 文件夹中总图片数量
        """
        try:
            # 读取JSON配置文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 获取图片处理选项
            img_proc_config = config.get("image_processing", {})
            
            # 转换为process_single_image函数所需的options格式
            options = {
                'watermark_text': img_proc_config.get("watermark", {}).get("text", ""),
                'brightness_factor': img_proc_config.get("brightness", {}).get("factor", 1.0),
                'contrast_factor': img_proc_config.get("contrast", {}).get("factor", 1.0),
                'saturation_factor': img_proc_config.get("saturation", {}).get("factor", 1.0),
                'sharpness_factor': img_proc_config.get("sharpness", {}).get("factor", 1.0),
                'blur_radius': img_proc_config.get("blur", {}).get("radius", 0.0),
                'resize_factor': img_proc_config.get("resize", {}).get("factor", 1.0),
                'crop_percentage': img_proc_config.get("crop", {}).get("percentage", 0.0),
                'add_border': img_proc_config.get("add_border", False),
                'add_noise': img_proc_config.get("add_noise", False),
                'remove_metadata': img_proc_config.get("remove_metadata", True),
                'change_md5': img_proc_config.get("change_md5", False),
                'ai_rewrite': img_proc_config.get("ai_rewrite", False)  # 添加AI改写选项
            }
            
            # 获取文件夹中的所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp']
            image_files = []
            
            for file in os.listdir(image_folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(image_folder_path, file))
            
            if not image_files:
                logger.warning(f"在文件夹 {image_folder_path} 中没有找到图片文件")
                return 0, 0
            
            # 处理每张图片并覆盖原图
            processed_count = 0
            for i, image_path in enumerate(image_files):
                try:
                    logger.info(f"正在处理图片 {i+1}/{len(image_files)}: {image_path}")
                    
                    # 处理单张图片，输出路径与输入路径相同（覆盖原图）
                    result_path = self.process_single_image(
                        image_path, 
                        image_path,  # 使用相同路径覆盖原图
                        options
                    )
                    
                    if result_path:
                        processed_count += 1
                        logger.info(f"✅ 图片处理完成: {result_path}")
                    else:
                        logger.error(f"❌ 图片处理失败: {image_path}")
                        
                except Exception as e:
                    logger.error(f"处理图片时出错 {image_path}: {str(e)}")
            
            logger.info(f"图片处理完成，共处理 {processed_count}/{len(image_files)} 张图片")
            return processed_count, len(image_files)
            
        except Exception as e:
            logger.error(f"从JSON配置处理图片时出错: {str(e)}")
            return 0, 0

    def process_single_image(self, input_path, output_path=None, options=None):
        """
        处理单张图片：亮度调节、尺寸调整、裁剪、添加水印、添加边框、删除元数据等

        参数:
        - input_path: 输入图片路径
        - output_path: 输出图片路径（可选，默认为原文件名_processed.原格式）
        - options: 处理选项字典
        """
        try:
            # 默认选项
            if options is None:
                options = {
                    'watermark_text': "Watermark",
                    'brightness_factor': 1.1,
                    'contrast_factor': 1.0,
                    'saturation_factor': 1.0,
                    'sharpness_factor': 1.0,
                    'blur_radius': 0.0,
                    'resize_factor': 1.0,
                    'crop_percentage': 0.0,
                    'add_border': False,
                    'add_noise': False,
                    'mirror_horizontal': False,
                    'mirror_vertical': False,
                    'remove_metadata': True,
                    'change_md5': False,
                    # 新增功能选项
                    'hue_factor': 0.0,
                    'color_temp_factor': 0.0,
                    'red_balance': 1.0,
                    'green_balance': 1.0,
                    'blue_balance': 1.0,
                    'convert_to_grayscale': False,
                    'grayscale_method': 'weighted',
                    'shadow_factor': 0.0,
                    'highlight_factor': 0.0,
                    'local_brightness_center_x': 0.5,
                    'local_brightness_center_y': 0.5,
                    'local_brightness_radius': 0.3,
                    'local_brightness_factor': 1.0,
                    'gradient_type': None,
                    'gradient_direction': 0,
                    'gradient_intensity': 0.3,
                    'exposure_factor': 0.0,
                    'smart_sharpen_amount': 0.0,
                    'smart_sharpen_radius': 1.0,
                    'smart_sharpen_threshold': 0,
                    'reduce_noise_strength': 0.0,
                    'grain_size': 0.5,
                    'grain_strength': 0.0,
                    'enhance_details_strength': 0.0,
                    'rotation_angle': 0.0,
                    'perspective_distortion': 0.0,
                    'distortion_type': None,
                    'distortion_strength': 0.0,
                    'deformation_center_x': 0.5,
                    'deformation_center_y': 0.5,
                    'deformation_radius': 0.2,
                    'deformation_strength': 0.0,
                    'vignette_strength': 0.0,
                    'chromatic_aberration': 0.0,
                    'soft_glow_strength': 0.0,
                    'texture_type': None,
                    'texture_intensity': 0.0
                }
            
            # 打开原始图片
            img = Image.open(input_path)
            original_format = img.format  # 获取原始格式

            # 如果没有指定输出路径，自动生成
            if output_path is None:
                file_dir = os.path.dirname(input_path)
                file_name, file_ext = os.path.splitext(os.path.basename(input_path))
                output_path = os.path.join(file_dir, f"{file_name}_processed{file_ext}")

            logger.info(f"原始格式: {original_format}")
            logger.info(f"输入路径: {input_path}")
            logger.info(f"输出路径: {output_path}")

            # 确保图片是RGB模式（用于后续处理）
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')

            # 保存原始的透明度信息
            has_transparency = img.mode == 'RGBA' or 'transparency' in img.info

            # 1. 几何变换（最先处理，避免影响后续效果）
            
            # 1.1 轻微旋转
            rotation_angle = options.get('rotation_angle', 0.0)
            if rotation_angle != 0.0:
                img = self.slight_rotation(img, rotation_angle)
                logger.info(f"轻微旋转: {rotation_angle}度")
            
            # 1.2 透视调整
            perspective_distortion = options.get('perspective_distortion', 0.0)
            if perspective_distortion != 0.0:
                img = self.adjust_perspective(img, perspective_distortion)
                logger.info(f"透视调整: {perspective_distortion}")
            
            # 1.3 轻微扭曲
            distortion_type = options.get('distortion_type', None)
            if distortion_type:
                distortion_strength = options.get('distortion_strength', 0.0)
                img = self.slight_distortion(img, distortion_type, distortion_strength)
                logger.info(f"轻微扭曲: {distortion_type}, 强度: {distortion_strength}")
            
            # 1.4 局部变形
            deformation_strength = options.get('deformation_strength', 0.0)
            if deformation_strength != 0.0:
                deformation_center_x = options.get('deformation_center_x', 0.5)
                deformation_center_y = options.get('deformation_center_y', 0.5)
                deformation_radius = options.get('deformation_radius', 0.2)
                img = self.local_deformation(img, deformation_center_x, deformation_center_y, 
                                           deformation_radius, deformation_strength)
                logger.info(f"局部变形: 中心({deformation_center_x}, {deformation_center_y}), 半径: {deformation_radius}, 强度: {deformation_strength}")

            # 2. 色彩调整
            
            # 2.1 色相调整
            hue_factor = options.get('hue_factor', 0.0)
            if hue_factor != 0.0:
                img = self.adjust_hue(img, hue_factor)
                logger.info(f"色相调整: {hue_factor}")
            
            # 2.2 色温调整
            color_temp_factor = options.get('color_temp_factor', 0.0)
            if color_temp_factor != 0.0:
                img = self.adjust_color_temperature(img, color_temp_factor)
                logger.info(f"色温调整: {color_temp_factor}")
            
            # 2.3 色彩平衡
            red_balance = options.get('red_balance', 1.0)
            green_balance = options.get('green_balance', 1.0)
            blue_balance = options.get('blue_balance', 1.0)
            if red_balance != 1.0 or green_balance != 1.0 or blue_balance != 1.0:
                img = self.adjust_color_balance(img, red_balance, green_balance, blue_balance)
                logger.info(f"色彩平衡: R={red_balance}, G={green_balance}, B={blue_balance}")
            
            # 2.4 黑白转换
            convert_to_grayscale = options.get('convert_to_grayscale', False)
            if convert_to_grayscale:
                grayscale_method = options.get('grayscale_method', 'weighted')
                img = self.convert_to_grayscale(img, grayscale_method)
                logger.info(f"黑白转换: {grayscale_method}")

            # 3. 光影效果
            
            # 3.1 阴影/高光调整
            shadow_factor = options.get('shadow_factor', 0.0)
            highlight_factor = options.get('highlight_factor', 0.0)
            if shadow_factor != 0.0 or highlight_factor != 0.0:
                img = self.adjust_shadows_highlights(img, shadow_factor, highlight_factor)
                logger.info(f"阴影/高光调整: 阴影={shadow_factor}, 高光={highlight_factor}")
            
            # 3.2 局部亮度调整
            local_brightness_factor = options.get('local_brightness_factor', 1.0)
            if local_brightness_factor != 1.0:
                local_brightness_center_x = options.get('local_brightness_center_x', 0.5)
                local_brightness_center_y = options.get('local_brightness_center_y', 0.5)
                local_brightness_radius = options.get('local_brightness_radius', 0.3)
                img = self.adjust_local_brightness(img, local_brightness_center_x, local_brightness_center_y,
                                                local_brightness_radius, local_brightness_factor)
                logger.info(f"局部亮度调整: 中心({local_brightness_center_x}, {local_brightness_center_y}), 半径: {local_brightness_radius}, 因子: {local_brightness_factor}")
            
            # 3.3 渐变映射
            gradient_type = options.get('gradient_type', None)
            if gradient_type:
                gradient_direction = options.get('gradient_direction', 0)
                gradient_intensity = options.get('gradient_intensity', 0.3)
                img = self.apply_gradient_map(img, gradient_type, gradient_direction, gradient_intensity)
                logger.info(f"渐变映射: {gradient_type}, 方向: {gradient_direction}, 强度: {gradient_intensity}")
            
            # 3.4 曝光调整
            exposure_factor = options.get('exposure_factor', 0.0)
            if exposure_factor != 0.0:
                img = self.adjust_exposure(img, exposure_factor)
                logger.info(f"曝光调整: {exposure_factor}")

            # 4. 原有处理
            
            # 4.1 裁剪图片（上下左右各裁掉指定百分比）
            crop_percentage = options.get('crop_percentage', 0.0)
            if crop_percentage > 0:
                width, height = img.size
                left = int(width * crop_percentage)
                top = int(height * crop_percentage)
                right = int(width * (1 - crop_percentage))
                bottom = int(height * (1 - crop_percentage))
                img = img.crop((left, top, right, bottom))
                logger.info(f"裁剪图片: 上下左右各裁掉 {crop_percentage*100}%")

            # 4.2 调节对比度
            contrast_factor = options.get('contrast_factor', 1.0)
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
                logger.info(f"对比度调节: {contrast_factor}")

            # 4.3 调节饱和度
            saturation_factor = options.get('saturation_factor', 1.0)
            if saturation_factor != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation_factor)
                logger.info(f"饱和度调节: {saturation_factor}")

            # 4.4 调节亮度
            brightness_factor = options.get('brightness_factor', 1.1)
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)
                logger.info(f"亮度调节: {brightness_factor}")

            # 5. 细节处理
            
            # 5.1 智能锐化
            smart_sharpen_amount = options.get('smart_sharpen_amount', 0.0)
            if smart_sharpen_amount > 0.0:
                smart_sharpen_radius = options.get('smart_sharpen_radius', 1.0)
                smart_sharpen_threshold = options.get('smart_sharpen_threshold', 0)
                img = self.smart_sharpen(img, smart_sharpen_amount, smart_sharpen_radius, smart_sharpen_threshold)
                logger.info(f"智能锐化: 强度={smart_sharpen_amount}, 半径={smart_sharpen_radius}, 阈值={smart_sharpen_threshold}")
            
            # 5.2 噪点减少
            reduce_noise_strength = options.get('reduce_noise_strength', 0.0)
            if reduce_noise_strength > 0.0:
                img = self.reduce_noise(img, reduce_noise_strength)
                logger.info(f"噪点减少: 强度={reduce_noise_strength}")
            
            # 5.3 颗粒效果
            grain_strength = options.get('grain_strength', 0.0)
            if grain_strength > 0.0:
                grain_size = options.get('grain_size', 0.5)
                img = self.add_grain_effect(img, grain_size, grain_strength)
                logger.info(f"颗粒效果: 大小={grain_size}, 强度={grain_strength}")
            
            # 5.4 细节增强
            enhance_details_strength = options.get('enhance_details_strength', 0.0)
            if enhance_details_strength > 0.0:
                img = self.enhance_details(img, enhance_details_strength)
                logger.info(f"细节增强: 强度={enhance_details_strength}")

            # 5.5 原有锐化处理
            sharpness_factor = options.get('sharpness_factor', 1.0)
            if sharpness_factor != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpness_factor)
                logger.info(f"锐化处理: {sharpness_factor}")

            # 5.6 模糊处理
            blur_radius = options.get('blur_radius', 0.0)
            if blur_radius > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                logger.info(f"高斯模糊: 半径 {blur_radius}")

            # 6. 特殊效果
            
            # 6.1 晕影效果
            vignette_strength = options.get('vignette_strength', 0.0)
            if vignette_strength > 0.0:
                img = self.add_vignette(img, vignette_strength)
                logger.info(f"晕影效果: 强度={vignette_strength}")
            
            # 6.2 色差效果
            chromatic_aberration = options.get('chromatic_aberration', 0.0)
            if chromatic_aberration > 0.0:
                img = self.add_chromatic_aberration(img, chromatic_aberration)
                logger.info(f"色差效果: 强度={chromatic_aberration}")
            
            # 6.3 柔光效果
            soft_glow_strength = options.get('soft_glow_strength', 0.0)
            if soft_glow_strength > 0.0:
                img = self.add_soft_glow(img, soft_glow_strength)
                logger.info(f"柔光效果: 强度={soft_glow_strength}")
            
            # 6.4 纹理叠加
            texture_type = options.get('texture_type', None)
            if texture_type:
                texture_intensity = options.get('texture_intensity', 0.0)
                img = self.add_texture_overlay(img, texture_type, texture_intensity)
                logger.info(f"纹理叠加: 类型={texture_type}, 强度={texture_intensity}")

            # 7. 尺寸和裁剪
            
            # 7.1 微调尺寸
            resize_factor = options.get('resize_factor', 1.0)
            if resize_factor != 1.0:
                new_size = tuple(int(dim * resize_factor) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                logger.info(f"尺寸调整: {resize_factor}")

            # 8. 水印和边框
            
            # 8.1 添加噪点
            if options.get('add_noise', False):
                img = self.add_noise(img)
                logger.info("添加噪点")

            # 8.2 添加边框
            if options.get('add_border', False):
                # 添加2%大小的边框
                border_size = max(1, int(min(img.size) * 0.02))
                img = ImageOps.expand(img, border=border_size, fill='black')
                logger.info(f"添加边框: {border_size}px")
            
            # 8.3 镜像反转
            if options.get('mirror_horizontal', False):
                img = ImageOps.mirror(img)
                logger.info("左右反转（水平镜像）")
            
            if options.get('mirror_vertical', False):
                img = ImageOps.flip(img)
                logger.info("上下反转（垂直镜像）")

            # 8.4 添加透明水印
            watermark_text = options.get('watermark_text', "Watermark")
            if watermark_text:  # 只有当水印文字不为空时才添加
                # 转换为RGBA模式以支持透明度
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # 创建透明图层
                txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
                draw_txt = ImageDraw.Draw(txt_layer)

                # 获取字体（优先使用中文字体）
                font = self.get_chinese_font(24)
                
                # 尝试获取文本边界框，用于定位
                try:
                    # 使用textbbox获取文本尺寸
                    bbox = draw_txt.textbbox((0, 0), watermark_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    # 如果textbbox不可用，使用旧方法
                    try:
                        text_width, text_height = draw_txt.textsize(watermark_text, font)
                    except:
                        # 如果所有方法都失败，使用默认尺寸
                        text_width, text_height = 100, 20

                # 计算水印位置（右下角）
                margin = 10
                x = img.width - text_width - margin
                y = img.height - text_height - margin

                # 添加半透明文字（使用中文支持的颜色）
                draw_txt.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))

                # 合并图层
                img = Image.alpha_composite(img, txt_layer)
                logger.info(f"添加水印: {watermark_text}")

            # 9. 元数据处理

            # 9.1 根据输出路径确定保存格式
            output_ext = os.path.splitext(output_path)[1].lower()

            # 格式映射
            format_map = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.bmp': 'BMP',
                '.gif': 'GIF',
                '.tiff': 'TIFF',
                '.tif': 'TIFF',
                '.webp': 'WEBP'
            }

            # 确定保存格式
            if output_ext in format_map:
                save_format = format_map[output_ext]
            else:
                # 如果扩展名不在映射中，使用原始格式
                save_format = original_format if original_format else 'PNG'

            # 处理不同格式的特殊要求
            if save_format == 'JPEG':
                # JPEG不支持透明度，需要转换为RGB
                if img.mode == 'RGBA':
                    # 创建白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

            elif save_format == 'PNG':
                # PNG保持RGBA模式以支持透明度
                if not has_transparency and img.mode == 'RGBA':
                    # 如果原图没有透明度但现在是RGBA，可以选择转换为RGB
                    pass  # 保持RGBA以支持水印透明度

            # 9.2 删除元数据或保留元数据
            if options.get('remove_metadata', True):
                # 创建新图片以删除元数据
                img_no_exif = Image.new(img.mode, img.size)
                img_no_exif.putdata(list(img.getdata()))
                img = img_no_exif
                logger.info("删除元数据")

            # 10. 保存图片
            save_kwargs = {'optimize': True}
            if save_format == 'JPEG':
                save_kwargs['quality'] = 95  # 设置JPEG质量

            img.save(output_path, save_format, **save_kwargs)
            logger.info(f"图片处理完成！保存格式: {save_format}")
            
            return output_path

        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
            return None


# 测试代码
if __name__ == "__main__":
    # 创建图片处理器实例
    processor = ImageProcessor()
    
    # 测试图片路径
    test_image_path = "e:\\项目课程\\RPA开发\\客户项目\\solar\\定制-图片去重批量处理\\1.jpg"
    
    # 处理选项
    options = {
        'watermark_text': "测试水印",
        'brightness_factor': 1.1,
        'contrast_factor': 1.1,
        'saturation_factor': 1.1,
        'sharpness_factor': 1.1,
        'blur_radius': 0.0,
        'resize_factor': 1.0,
        'crop_percentage': 0.0,
        'add_border': True,
        'add_noise': False,
        'mirror_horizontal': True,  # 添加水平镜像反转
        'mirror_vertical': False,
        'remove_metadata': True,
        'change_md5': False
    }
    
    # 处理单张图片
    result_path = processor.process_single_image(test_image_path, options=options)
    
    if result_path:
        print(f"图片处理成功！输出路径: {result_path}")
    else:
        print("图片处理失败！")